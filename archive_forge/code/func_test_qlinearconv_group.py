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
def test_qlinearconv_group(self) -> None:
    graph = self._make_graph([('x', TensorProto.INT8, (30, 4, 8, 8, 8)), ('x_scale', TensorProto.FLOAT, ()), ('x_zero_point', TensorProto.INT8, ()), ('w', TensorProto.INT8, (4, 1, 8, 8, 8)), ('w_scale', TensorProto.FLOAT, ()), ('w_zero_point', TensorProto.INT8, ()), ('y_scale', TensorProto.FLOAT, ()), ('y_zero_point', TensorProto.INT8, ())], [make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], 'y', group=4)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT8, (30, 4, 1, 1, 1))])