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
def _prepare_input_for_onnx(args, kwargs, remained_onnx_input_idx: Optional[Sequence[int]], flatten: bool):
    """Prepare input for ONNX model execution in ONNX backend.

    Any future changes/formatting to the input before dispatching to the ONNX backend
    run should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.
        remained_onnx_input_idx: indices of inputs to be used for ONNX model execution.
        flatten: whether to flatten the input before dispatching to the ONNX model execution.

    Returns:
        onnx_inputs: positional arguments for ONNX model execution in ONNX backend.
    """
    onnx_inputs = _prepare_input_for_export(args, kwargs)
    if flatten:
        onnx_inputs, _ = torch.jit._flatten(onnx_inputs)
    elif onnx_inputs and onnx_inputs[-1] == {}:
        onnx_inputs = onnx_inputs[:-1]
    if remained_onnx_input_idx is not None:
        return [onnx_inputs[i] for i in remained_onnx_input_idx]
    else:
        return onnx_inputs