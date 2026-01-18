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
def test_resize_size_not_smaller(self, _, version) -> None:
    self.skipIf(version < 18, 'keep_aspect_ratio_policy is from Version 18')
    graph = self._make_graph([('x', TensorProto.INT32, (3, 5)), ('roi', TensorProto.FLOAT, (4,)), ('sizes', TensorProto.INT64, (2,))], [make_node('Resize', ['x', 'roi', '', 'sizes'], ['y'], keep_aspect_ratio_policy='not_smaller')], [], initializer=[make_tensor('sizes', TensorProto.INT64, (2,), (6, 6))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, (6, 10))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])