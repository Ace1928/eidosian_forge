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
class OnnxBackend(enum.Enum):
    """Enum class for ONNX backend used for export verification."""
    REFERENCE = 'ONNXReferenceEvaluator'
    ONNX_RUNTIME_CPU = 'CPUExecutionProvider'
    ONNX_RUNTIME_CUDA = 'CUDAExecutionProvider'