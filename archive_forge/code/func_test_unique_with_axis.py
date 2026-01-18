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
def test_unique_with_axis(self) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, (2, 4, 2))], [make_node('Unique', ['X'], ['Y', 'indices', 'inverse_indices', 'counts'], axis=1)], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (2, None, 2)), make_tensor_value_info('indices', TensorProto.INT64, (None,)), make_tensor_value_info('inverse_indices', TensorProto.INT64, (None,)), make_tensor_value_info('counts', TensorProto.INT64, (None,))])