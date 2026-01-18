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
@parameterized.expand(all_versions_for('Reshape'))
def test_reshape_static_shape_constant(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.UINT8, (2, 4, 3))], [make_node('Constant', [], ['shape'], value=make_tensor('shape', TensorProto.INT64, (2,), (3, 8))), make_node('Reshape', ['x', 'shape'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('shape', TensorProto.INT64, (2,)), make_tensor_value_info('y', TensorProto.UINT8, (3, 8))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])