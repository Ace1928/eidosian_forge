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
def test_quantizelinear_zp_output_dtype_conflicted(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 4, 5)), ('y_scale', TensorProto.FLOAT, ()), ('y_zero_point', TensorProto.UINT16, ())], [make_node('QuantizeLinear', ['x', 'y_scale', 'y_zero_point'], ['y'], output_dtype=TensorProto.INT4)], [])
    self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)