from __future__ import annotations
import inspect
import logging
import operator
import re
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
import torch
import torch.fx
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
def _retrieve_or_adapt_input_to_graph_set(fx_node_arg: fx_type_utils.Argument, fx_name_to_onnxscript_value: Dict[str, Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]], tracer: onnxscript_graph_building.TorchScriptTracingEvaluator):
    """Map FX value to TorchScript value.

    When creating TorchScript graph from FX graph, we need a mapping from FX variable
    to TorchScript variable. This function maps FX variable, fx_node_arg, to torch.jit.Value.
    """
    onnx_tensor = fx_node_arg
    if isinstance(onnx_tensor, torch.fx.Node):
        return fx_name_to_onnxscript_value[onnx_tensor.name]
    if isinstance(onnx_tensor, (tuple, list)) and any((isinstance(node, torch.fx.Node) and fx_type_utils.is_torch_symbolic_type(node.meta.get('val')) for node in onnx_tensor)):
        sequence_mixed_elements: List[Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...], List[int]]] = []
        for tensor in onnx_tensor:
            if isinstance(tensor, torch.fx.Node) and fx_type_utils.is_torch_symbolic_type(tensor.meta.get('val')):
                sequence_mixed_elements.append(fx_name_to_onnxscript_value[tensor.name])
            elif isinstance(tensor, int):
                sequence_mixed_elements.append([tensor])
        with onnxscript.evaluator.default_as(tracer):
            output = onnxscript.opset18.Concat(*sequence_mixed_elements, axis=0)
        output.dtype = torch.int64
        output.shape = [len(sequence_mixed_elements)]
        return output
    elif isinstance(onnx_tensor, (tuple, list)) and all((isinstance(node, torch.fx.Node) or node is None for node in onnx_tensor)):
        sequence_elements: List[Union[Optional[onnxscript_graph_building.TorchScriptTensor], Tuple[onnxscript_graph_building.TorchScriptTensor, ...]]] = []
        for tensor in onnx_tensor:
            sequence_elements.append(fx_name_to_onnxscript_value[tensor.name] if tensor is not None else None)
        return sequence_elements
    if isinstance(onnx_tensor, torch.dtype):
        onnx_tensor = int(jit_type_utils.JitScalarType.from_dtype(onnx_tensor).onnx_type())
    if isinstance(onnx_tensor, torch.device):
        return str(onnx_tensor)
    return onnx_tensor