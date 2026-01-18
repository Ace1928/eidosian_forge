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
def test_reduce_op_shape_negative_axis(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (24, 4, 11)), ('axes', TensorProto.INT64, (2,))], [make_node('ReduceL1', ['x', 'axes'], 'y')], [], initializer=[make_tensor('axes', TensorProto.INT64, (2,), (-1, -2))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (24, 1, 1))])