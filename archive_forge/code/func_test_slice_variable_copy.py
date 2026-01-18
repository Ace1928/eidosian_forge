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
def test_slice_variable_copy(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, ('a', 2)), ('starts', TensorProto.INT64, (1,)), ('ends', TensorProto.INT64, (1,)), ('axes', TensorProto.INT64, (1,))], [make_node('Slice', ['x', 'starts', 'ends', 'axes'], 'y')], [], initializer=[make_tensor('starts', TensorProto.INT64, (1,), (1,)), make_tensor('ends', TensorProto.INT64, (1,), (200,)), make_tensor('axes', TensorProto.INT64, (1,), (1,))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ('a', 1))])