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
@parameterized.expand(all_versions_for('Resize'))
def test_resize_opset11_scales_is_empty(self, _, version) -> None:
    self.skipIf(version != 11, 'This test only works for Version 11')
    graph = self._make_graph([('x', TensorProto.INT32, (1, 3, 4, 5)), ('roi', TensorProto.FLOAT, (8,)), ('scales', TensorProto.FLOAT, (0,)), ('sizes', TensorProto.INT64, (4,))], [make_node('Resize', ['x', 'roi', 'scales', 'sizes'], ['y'])], [], initializer=[make_tensor('sizes', TensorProto.INT64, (4,), vals=np.array([2, 6, 8, 10], dtype='<i8').tobytes(), raw=True)])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, (2, 6, 8, 10))], opset_imports=[helper.make_opsetid('', version)])