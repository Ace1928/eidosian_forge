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
def _children_str_at_line(self, line: int) -> str:
    """Get the string representation of the children at the given line.

        Recursively calls `_str_at_line` on children nodes.
        """
    if self.upper_printer is None and self.lower_printer is None:
        return ''
    upper_total_rows = self.upper_printer._total_rows() if self.upper_printer else 1
    lower_total_rows = self.lower_printer._total_rows() if self.lower_printer else 1
    if 0 <= line < upper_total_rows:
        return self.upper_printer._str_at_line(line) if self.upper_printer else '...'
    elif upper_total_rows < line < upper_total_rows + lower_total_rows + 1:
        return self.lower_printer._str_at_line(line - upper_total_rows - 1) if self.lower_printer else '...'
    return ''