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
def test_slice_with_input_shape_steps(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (5, 6, 7)), ('starts', TensorProto.INT64, (3,)), ('ends', TensorProto.INT64, (3,)), ('axes', TensorProto.INT64, None), ('steps', TensorProto.INT64, (3,))], [make_node('Slice', ['x', 'starts', 'ends', 'axes', 'steps'], ['y'])], [], initializer=[make_tensor('starts', TensorProto.INT64, (3,), (1, 0, 0)), make_tensor('ends', TensorProto.INT64, (3,), (2, 6, 6)), make_tensor('steps', TensorProto.INT64, (3,), (1, 4, 3))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 2, 2))])