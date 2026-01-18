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
def test_space_to_depth_unknown_dim(self) -> None:
    b = 10
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 'N', 100, 100))], [make_node('SpaceToDepth', ['x'], ['z'], blocksize=b)], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (2, None, 10, 10))])