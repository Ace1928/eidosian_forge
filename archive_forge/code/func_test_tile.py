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
def test_tile(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (4, 5, 6)), ('repeats', TensorProto.INT64, (3,))], [make_node('Tile', ['x', 'repeats'], ['y'])], [], initializer=[make_tensor('repeats', TensorProto.INT64, (3,), (1, 2, 3))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (4, 10, 18))])