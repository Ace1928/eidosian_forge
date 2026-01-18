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
def test_split_with_split_attribute(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 4)), ('split', TensorProto.INT64, (2,))], [make_node('Split', ['x', 'split'], ['y', 'z'], axis=1)], [], initializer=[make_tensor('split', TensorProto.INT64, (2,), (3, 1))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (2, 3)), make_tensor_value_info('z', TensorProto.FLOAT, (2, 1))])