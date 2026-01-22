import torch
from torch.fx import map_arg, Node
from torch.fx.graph import Graph
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.quantized.reference as nnqr
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.fx import GraphModule
from .utils import (
from ..utils import _parent_name
from ..qconfig import QConfigAny
from ..quantization_mappings import get_quantized_operator
from .utils import create_node_from_old_node_preserve_meta
from typing import Dict, Tuple, Type, List, Callable, Any, Union, Set, Optional
import operator
def _lower_static_weighted_ref_functional(model: GraphModule, qconfig_map: Dict[str, QConfigAny]):
    """
    Traverse the graph and replace functional reference patterns with their quantized versions.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    nodes = list(model.graph.nodes)
    for n in model.graph.nodes:
        matching_ops = list(STATIC_LOWER_FUNCTIONAL_MAP.keys())
        q_node, relu_node, func_node = _match_static_pattern(n, modules, qconfig_map, matching_ops, dequantize_node_arg_indices=[0, 1])
        if q_node is None:
            continue
        assert func_node is not None
        _, output_scale_node, output_zp_node, _ = q_node.args
        input_dq_node, weight_dq_node, *remaining_func_args = func_node.args
        assert isinstance(output_zp_node, Node)
        assert isinstance(input_dq_node, Node)
        assert isinstance(weight_dq_node, Node)
        quantized_weight = weight_dq_node.args[0]
        assert isinstance(quantized_weight, Node)
        if quantized_weight.op != 'call_function' or quantized_weight.target not in (torch.quantize_per_tensor, torch.quantize_per_channel):
            continue
        prepack_args = [quantized_weight] + remaining_func_args
        if func_node.target == F.linear:
            weight_dtype = quantized_weight.args[-1]
            prepack_op = get_linear_prepack_op_for_dtype(weight_dtype)
        elif func_node.target in CONV_FUNCTIONAL_OPS:
            prepack_op = get_qconv_prepack_op(func_node.target)
            if func_node.target == F.conv1d:
                for i in [2, 3, 4]:
                    if len(prepack_args) > i and isinstance(prepack_args[i], int):
                        prepack_args[i] = (prepack_args[i],)
        elif func_node.target in CONV_TRANSPOSE_FUNCTIONAL_OPS:
            prepack_op = get_qconv_prepack_op(func_node.target)
            if func_node.target == F.conv_transpose1d:
                for i in [2, 3, 4, 6]:
                    if len(prepack_args) > i and isinstance(prepack_args[i], int):
                        prepack_args[i] = (prepack_args[i],)
            if len(prepack_args) > 6:
                prepack_args[5], prepack_args[6] = (prepack_args[6], prepack_args[5])
        else:
            raise ValueError(f"Lowering is not supported for op '{func_node.target}'")
        with model.graph.inserting_before(output_scale_node):
            kwargs = func_node.kwargs
            if func_node.target == F.linear and 'bias' in kwargs:
                kwargs = kwargs.copy()
                kwargs['B'] = kwargs['bias']
                del kwargs['bias']
            packed_weight = model.graph.create_node('call_function', prepack_op, tuple(prepack_args), kwargs)
        q_func, q_relu_func = STATIC_LOWER_FUNCTIONAL_MAP[func_node.target]
        if q_relu_func is not None:
            func_node.target = q_relu_func if relu_node is not None else q_func
        else:
            func_node.target = q_func
        func_node.args = (input_dq_node.args[0], packed_weight, output_scale_node, output_zp_node)
        func_node.kwargs = {}
        q_node.replace_all_uses_with(func_node)
        output_zp_node.append(func_node)
        model.graph.erase_node(q_node)
        if relu_node is not None and q_relu_func is not None:
            model.graph.erase_node(relu_node)