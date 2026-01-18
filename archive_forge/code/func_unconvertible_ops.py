from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@_beartype.beartype
def unconvertible_ops(model, args, training: _C_onnx.TrainingMode=_C_onnx.TrainingMode.EVAL, opset_version: Optional[int]=None) -> Tuple[_C.Graph, List[str]]:
    """Returns an approximated list of all ops that are yet supported by :mod:`torch.onnx`.

    The list is approximated because some ops may be removed during the conversion
    process and don't need to be converted. Some other ops may have partial support
    that will fail conversion with particular inputs. Please open a Github Issue
    for op support requests.

    Args:
        model: Same as the `model` parameter in :func:`torch.onnx.export`.
        args: Same as the `args` parameter in :func:`torch.onnx.export`.
        training: Same as the `training` parameter in :func:`torch.onnx.export`.
        opset_version: Same as the `opset_version` parameter in :func:`torch.onnx.export`.

    Returns:
        The JIT graph and a list of unconvertible ops in the format of "domain::op".
    """
    opset_version = opset_version or _constants.ONNX_DEFAULT_OPSET
    GLOBALS.export_onnx_opset_version = opset_version
    try:
        with exporter_context(model, training, verbose=False):
            args = _decide_input_format(model, args)
            model = _pre_trace_quant_model(model, args)
            graph, _, _, module = _create_jit_graph(model, args)
            _C._jit_pass_inline(graph)
            _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
            _C._jit_pass_erase_number_types(graph)
            _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    except Exception as e:
        raise errors.OnnxExporterError('Failed to discover unconvertible ops because of errors during the JIT graph generation process.') from e
    unsupported_ops = []
    for node in graph.nodes():
        domain_op = node.kind()
        if domain_op.startswith(('onnx::', 'prim::')):
            continue
        if not registration.registry.is_registered_op(domain_op.rstrip('_'), opset_version):
            unsupported_ops.append(domain_op)
    return (graph, unsupported_ops)