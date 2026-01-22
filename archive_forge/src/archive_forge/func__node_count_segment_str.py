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
def _node_count_segment_str(self) -> str:
    if self.graph_info is None:
        return '...'
    node_count = self.graph_info.essential_node_count()
    has_mismatch = self.graph_info.has_mismatch()
    error_node_kind = f'({self.graph_info.essential_node_kinds().pop()})' if node_count == 1 and has_mismatch else ''
    return f'{node_count} {('X' if has_mismatch else '✓')} {error_node_kind}'