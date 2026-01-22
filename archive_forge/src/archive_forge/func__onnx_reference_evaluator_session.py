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
def _onnx_reference_evaluator_session(model: Union[str, io.BytesIO]):
    try:
        import onnx
        from onnx import reference as onnx_reference
    except ImportError as exc:
        raise ImportError('onnx >= 1.13 is required for reference evaluator.') from exc
    proto = onnx.load(model) if isinstance(model, str) else onnx.load_model_from_string(model.getvalue())
    onnx_session = onnx_reference.ReferenceEvaluator(proto)
    return onnx_session