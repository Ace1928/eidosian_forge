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
def _run_onnx(onnx_session, inputs) -> _OutputsType:
    kw_inputs = {}
    if inputs and isinstance(inputs[-1], dict):
        kw_inputs = inputs[-1]
        inputs = inputs[:-1]
    inputs = _unpack_to_numpy(_flatten_tuples(inputs))
    ort_inputs = {}
    for input_name, input in kw_inputs.items():
        ort_inputs[input_name] = _to_numpy(input)
    inputs = _to_numpy(inputs)
    if hasattr(onnx_session, 'get_inputs'):
        input_names = [i.name for i in onnx_session.get_inputs()]
    elif hasattr(onnx_session, 'input_names'):
        input_names = onnx_session.input_names
    else:
        raise ValueError(f'Unknown ONNX backend type: {type(onnx_session)}.')
    for i, input in enumerate(inputs):
        if i == len(input_names) or input_names[i] in ort_inputs:
            raise ValueError(f'got too many positional inputs. inputs: {inputs}. kw_inputs: {kw_inputs}. input names: {input_names}.')
        ort_inputs[input_names[i]] = input
    onnx_outs = onnx_session.run(None, ort_inputs)
    return onnx_outs