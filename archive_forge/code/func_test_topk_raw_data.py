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
def test_topk_raw_data(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 4, 5, 10))], [make_node('TopK', ['x', 'k'], ['y', 'z'], axis=2)], [], initializer=[make_tensor('k', TensorProto.INT64, (1,), vals=np.array([3], dtype='<i8').tobytes(), raw=True)])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, 4, 3, 10)), make_tensor_value_info('z', TensorProto.INT64, (3, 4, 3, 10))])