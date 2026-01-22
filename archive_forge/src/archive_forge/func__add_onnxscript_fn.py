from __future__ import annotations
import glob
import io
import os
import shutil
import zipfile
from typing import Any, List, Mapping, Set, Tuple, Union
import torch
import torch.jit._trace
import torch.serialization
from torch.onnx import _constants, _exporter_states, errors
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def _add_onnxscript_fn(model_bytes: bytes, custom_opsets: Mapping[str, int]) -> bytes:
    """Insert model-included custom onnx-script function into ModelProto"""
    try:
        import onnx
    except ImportError as e:
        raise errors.OnnxExporterError('Module onnx is not installed!') from e
    model_proto = onnx.load_model_from_string(model_bytes)
    onnx_function_list = list()
    included_node_func = set()
    _find_onnxscript_op(model_proto.graph, included_node_func, custom_opsets, onnx_function_list)
    if onnx_function_list:
        model_proto.functions.extend(onnx_function_list)
        model_bytes = model_proto.SerializeToString()
    return model_bytes