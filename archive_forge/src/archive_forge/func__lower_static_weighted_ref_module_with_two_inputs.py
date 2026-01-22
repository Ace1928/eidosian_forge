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
def _lower_static_weighted_ref_module_with_two_inputs(model: GraphModule, qconfig_map: Dict[str, QConfigAny]):
    """
    Traverse the graph and find patterns
    dequantize   dequantize
       \\         //
        ref module
            \\
          quantize
    and replace them with the quantized version of the ref module.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    nodes = list(model.graph.nodes)
    for n in model.graph.nodes:
        matching_modules = list(STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP.keys())
        q_node, ref_node = _match_static_pattern_with_two_inputs(n, modules, qconfig_map, matching_modules)
        if q_node is None:
            continue
        assert ref_node is not None
        _, scale_node, zero_point_node, _ = q_node.args
        ref_module = _get_module(ref_node, modules)
        ref_class = type(ref_module)
        assert isinstance(scale_node, Node)
        assert isinstance(zero_point_node, Node)
        assert issubclass(ref_class, nn.Module)
        if ref_class in STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP:
            inner_ref_class, q_class = STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP[ref_class]
            if type(ref_module[0]) != inner_ref_class:
                continue
        else:
            continue
        output_scale = getattr(model, scale_node.target)
        output_zero_point = getattr(model, zero_point_node.target)
        q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(modules[parent_name], module_name, q_module)
        assert len(ref_node.args) == 2
        for arg in ref_node.args:
            if not is_dequantize_node(arg):
                continue
            dq_node = arg
            assert isinstance(dq_node, Node)
            ref_node.replace_input_with(dq_node, dq_node.args[0])
        q_node.replace_all_uses_with(ref_node)
        model.graph.erase_node(q_node)
        model.graph.erase_node(scale_node)
        model.graph.erase_node(zero_point_node)