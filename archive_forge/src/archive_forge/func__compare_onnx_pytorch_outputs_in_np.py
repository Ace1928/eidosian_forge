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
def _compare_onnx_pytorch_outputs_in_np(onnx_outs: _OutputsType, pt_outs: _OutputsType, options: VerificationOptions):
    assert len(onnx_outs) == len(pt_outs), f'Number of outputs differ ONNX runtime: ({len(onnx_outs)}) PyTorch: ({len(pt_outs)})'
    acceptable_error_percentage = options.acceptable_error_percentage
    if acceptable_error_percentage and (acceptable_error_percentage > 1.0 or acceptable_error_percentage < 0.0):
        raise ValueError('If set, acceptable_error_percentage should be between 0.0 and 1.0')
    for ort_out, pt_out in zip(onnx_outs, pt_outs):
        try:
            if not options.check_shape:
                ort_out, pt_out = np.broadcast_arrays(ort_out, pt_out)
            torch.testing.assert_close(ort_out, pt_out, rtol=options.rtol, atol=options.atol, check_dtype=options.check_dtype, equal_nan=True)
        except AssertionError as e:
            if acceptable_error_percentage:
                error_percentage = 1 - np.sum(np.isclose(ort_out, pt_out, rtol=options.rtol, atol=options.atol)) / np.prod(ort_out.shape)
                if error_percentage <= acceptable_error_percentage:
                    warnings.warn(f'Suppressed AssertionError:\n{e}.\nError percentage {error_percentage} within acceptable range {acceptable_error_percentage}.')
                    continue
            if ort_out.dtype == np.uint8 or ort_out.dtype == np.int8:
                warnings.warn('ONNX output is quantized')
            if pt_out.dtype == np.uint8 or pt_out.dtype == np.int8:
                warnings.warn('PyTorch output is quantized')
            raise