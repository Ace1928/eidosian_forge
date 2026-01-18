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
def test_slice_with_input_shape_axes(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 6, 2)), ('starts', TensorProto.INT64, (2,)), ('ends', TensorProto.INT64, (2,)), ('axes', TensorProto.INT64, (2,)), ('steps', TensorProto.INT64, None)], [make_node('Slice', ['x', 'starts', 'ends', 'axes', 'steps'], ['y'])], [], initializer=[make_tensor('starts', TensorProto.INT64, (2,), (1, 0)), make_tensor('ends', TensorProto.INT64, (2,), (2, 2)), make_tensor('axes', TensorProto.INT64, (2,), (0, 2))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 6, 2))])