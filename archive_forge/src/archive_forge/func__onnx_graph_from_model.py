from __future__ import annotations
import contextlib
import copy
import dataclasses
import datetime
import difflib
import enum
import functools
import io
import itertools
import os
import tempfile
import warnings
from typing import (
import numpy as np
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _experimental, _exporter_states, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, onnx_proto_utils
from torch.types import Number
@_beartype.beartype
def _onnx_graph_from_model(model: Union[torch.nn.Module, torch.jit.ScriptModule], args: Tuple[Any, ...], kwargs: Mapping[str, Any], export_options: _experimental.ExportOptions) -> _C.Graph:
    """As part of the ONNX export steps, export an ONNX JIT graph from a PyTorch model.

    Args:
        model: See :func:`check_export_model_diff`.
        args: See :func:`check_export_model_diff`.
        kwargs: See :func:`check_export_model_diff`.
        export_options: See :func:`check_export_model_diff`.

    Returns:
        onnx_graph (_C.Graph): An ONNX JIT graph.
    """
    opset_version = export_options.opset_version
    operator_export_type = export_options.operator_export_type
    export_modules_as_functions = export_options.export_modules_as_functions
    training = export_options.training
    verbose = export_options.verbose
    dynamic_axes = export_options.dynamic_axes
    input_names = export_options.input_names
    output_names = export_options.output_names
    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET
    utils._setup_trace_module_map(model, export_modules_as_functions)
    if not operator_export_type:
        if _C_onnx._CAFFE2_ATEN_FALLBACK:
            operator_export_type = _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        else:
            operator_export_type = _C_onnx.OperatorExportTypes.ONNX
    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_export_type
    with utils.exporter_context(model, training, verbose):
        do_constant_folding = utils._decide_constant_folding(export_options.do_constant_folding, operator_export_type, training)
        if dynamic_axes is None:
            dynamic_axes = {}
        utils._validate_dynamic_axes(dynamic_axes, model, input_names, output_names)
        export_inputs = _prepare_input_for_export(args, kwargs)
        export_inputs = utils._decide_input_format(model, export_inputs)
        onnx_graph, _, _ = utils._model_to_graph(model, export_inputs, verbose, input_names, output_names, operator_export_type, do_constant_folding, training=training, dynamic_axes=dynamic_axes)
        return onnx_graph