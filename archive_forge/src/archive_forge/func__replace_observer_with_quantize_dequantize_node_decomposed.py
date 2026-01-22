from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable
from torch.ao.quantization.quant_type import QuantType
import torch
import copy
import warnings
from torch.fx import (
from torch.fx.graph import (
from ..utils import (
from ..qconfig import (
from ..qconfig_mapping import QConfigMapping
from .qconfig_mapping_utils import (
from torch.ao.quantization.backend_config.utils import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.observer import _is_activation_post_process
from .graph_module import (
from ._equalize import update_obs_for_equalization, convert_eq_obs
from torch.nn.utils.parametrize import type_before_parametrizations
from .utils import (
from torch.ao.quantization.utils import (
from torch.ao.quantization.quantize import (
from torch.ao.quantization.stubs import DeQuantStub
from .custom_config import (
from .lower_to_fbgemm import lower_to_fbgemm
from ._decomposed import quantized_decomposed_lib  # noqa: F401
import operator
def _replace_observer_with_quantize_dequantize_node_decomposed(model: torch.nn.Module, graph: Graph, node: Node, modules: Dict[str, torch.nn.Module], node_name_to_scope: Dict[str, Tuple[str, type]], node_name_to_qconfig: Dict[str, QConfigAny]) -> None:
    """ Replace activation_post_process module call node with quantize and
    dequantize node working with decomposed Tensor

    Before:
    ... -> observer_0(x) -> ...
    After:
    ... -> torch.ops.quantized_decomposed.quantize_per_tensor(x, ...) ->
    torch.ops.quantized_decomposed.dequantize_per_tensor() -> ...

    or quantize_per_channel and dequantize_per_channel
    """
    assert modules is not None
    assert isinstance(node.target, str)
    module_path, prefix = _get_module_path_and_prefix(node, node_name_to_scope, node_name_to_qconfig)
    activation_post_process = modules[node.target]
    skip_replacement = all((_has_none_qconfig(n, node_name_to_qconfig) for n in list(node.args) + list(node.users.keys())))
    if skip_replacement or not _is_conversion_supported(activation_post_process):
        with graph.inserting_before(node):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
        return
    dtype = activation_post_process.dtype
    is_dynamic = False
    if hasattr(activation_post_process, 'is_dynamic'):
        is_dynamic = activation_post_process.is_dynamic
    if dtype in [torch.quint8, torch.qint8, torch.qint32, torch.uint8, torch.int8, torch.int16, torch.int32] and (not is_dynamic):
        node_type = 'call_function'
        quantize_op: Optional[Callable] = None
        scale, zero_point = activation_post_process.calculate_qparams()
        if is_per_channel(activation_post_process.qscheme):
            ch_axis = int(activation_post_process.ch_axis)
            quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default
            quant_min = activation_post_process.quant_min
            quant_max = activation_post_process.quant_max
            dtype_ = to_underlying_dtype(dtype)
            qparams = {'_scale_': scale, '_zero_point_': zero_point, '_axis_': ch_axis, '_quant_min_': quant_min, '_quant_max_': quant_max, '_dtype_': dtype_}
        else:
            quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            scale = float(scale)
            zero_point = int(zero_point)
            quant_min = activation_post_process.quant_min
            quant_max = activation_post_process.quant_max
            dtype_ = to_underlying_dtype(dtype)
            qparams = {'_scale_': scale, '_zero_point_': zero_point, '_quant_min_': quant_min, '_quant_max_': quant_max, '_dtype_': dtype_}
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                if key in ['_scale_', '_zero_point_'] and (not isinstance(value_or_node, (float, int))):
                    qparam_node = create_getattr_from_value(model, graph, module_path + prefix + key, value_or_node)
                    quantize_op_inputs.append(qparam_node)
                else:
                    quantize_op_inputs.append(value_or_node)
            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            dq_inputs = [quantized_node] + quantize_op_inputs[1:]
            dequantized_node = graph.call_function(dequantize_op, tuple(dq_inputs), {})
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)
    elif is_dynamic:
        node_type = 'call_function'
        quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.tensor
        dtype_ = to_underlying_dtype(dtype)
        assert dtype_ in [torch.uint8, torch.int8], 'only uint8 and int8 are supported in reference flow for dynamic quantization right now'
        quant_min = activation_post_process.quant_min
        quant_max = activation_post_process.quant_max
        qscheme = getattr(activation_post_process, 'qscheme', torch.per_tensor_affine)
        eps = getattr(activation_post_process, 'eps', torch.finfo(torch.float32).eps)
        qparams = {'_quant_min_': quant_min, '_quant_max_': quant_max, '_eps_': eps, '_dtype_': dtype_}
        choose_qparams_op = _QSCHEME_TO_CHOOSE_QPARAMS_OP[qscheme]
        with graph.inserting_before(node):
            input_node = node.args[0]
            choose_qparams_op_inputs = [node.args[0]]
            for key, value in qparams.items():
                choose_qparams_op_inputs.append(value)
            choose_qparams_node = graph.create_node('call_function', choose_qparams_op, tuple(choose_qparams_op_inputs), {})
            scale_node = graph.create_node('call_function', operator.getitem, (choose_qparams_node, 0), {})
            zero_point_node = graph.create_node('call_function', operator.getitem, (choose_qparams_node, 1), {})
            quant_min = qparams['_quant_min_']
            quant_max = qparams['_quant_max_']
            dtype = qparams['_dtype_']
            qparams = {'_scale_': scale_node, '_zero_point_': zero_point_node, '_quant_min_': quant_min, '_quant_max_': quant_max, '_dtype_': dtype}
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                if key in ['_scale_', '_zero_point_']:
                    qparam_node = value_or_node
                    quantize_op_inputs.append(qparam_node)
                else:
                    quantize_op_inputs.append(value_or_node)
            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            dq_inputs = [quantized_node] + quantize_op_inputs[1:]
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor
            dequantized_node = graph.call_function(dequantize_op, tuple(dq_inputs), {})
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)
    elif dtype == torch.float16:
        raise NotImplementedError('decomposed to float16 op not implemented yet')