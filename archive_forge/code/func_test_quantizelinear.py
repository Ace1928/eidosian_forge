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
@parameterized.expand([onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16])
def test_quantizelinear(self, elem_type) -> None:
    graph = self._make_graph([('x', elem_type, (30, 4, 5)), ('y_scale', elem_type, ()), ('y_zero_point', TensorProto.UINT8, ())], [make_node('QuantizeLinear', ['x', 'y_scale', 'y_zero_point'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, (30, 4, 5))])