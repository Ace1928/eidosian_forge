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
def test_sum_multi_broadcasting(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (30, 1, 5)), ('y', TensorProto.FLOAT, ('a', 4, 1)), ('z', TensorProto.FLOAT, (4, 'b'))], [make_node('Sum', ['x', 'y', 'z'], ['out'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (30, 4, 5))])