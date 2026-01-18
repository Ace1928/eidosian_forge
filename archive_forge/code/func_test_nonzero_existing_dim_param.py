from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def test_nonzero_existing_dim_param(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3,))], [make_node('NonZero', ['x'], ['y'])], [make_tensor_value_info('y', TensorProto.INT64, (None, 'NZ'))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (1, 'NZ'))])