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
def test_slice_with_input_shape_containing_dim_params(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (1, 'a', 1)), ('starts', TensorProto.INT64, (3,)), ('ends', TensorProto.INT64, (3,))], [make_node('Slice', ['x', 'starts', 'ends'], ['y'])], [], initializer=[make_tensor('starts', TensorProto.INT64, (3,), (0, 0, 0)), make_tensor('ends', TensorProto.INT64, (3,), (1, 1, 1))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, None, 1))])