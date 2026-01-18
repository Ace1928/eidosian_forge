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
def test_resize_scale_and_size_but_one_is_empty(self, _, version) -> None:
    self.skipIf(version < 11, 'roi input is from Version 11')
    graph = self._make_graph([('x', TensorProto.INT32, (1, 3, 4, 5)), ('roi', TensorProto.FLOAT, (8,)), ('scales', TensorProto.FLOAT, (4,)), ('sizes', TensorProto.INT64, (0,))], [make_node('Resize', ['x', 'roi', 'scales', 'sizes'], ['y'])], [], initializer=[make_tensor('scales', TensorProto.FLOAT, (4,), vals=np.array([2.0, 1.1, 2.3, 1.9], dtype='<f4').tobytes(), raw=True), make_tensor('sizes', TensorProto.INT64, (0,), vals=np.array([], dtype='<i8').tobytes(), raw=True)])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, (2, 3, 9, 9))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])