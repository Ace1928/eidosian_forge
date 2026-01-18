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
@parameterized.expand(all_versions_for('Shape'))
def test_shape_start_1(self, _, version) -> None:
    self.skipIf(version < 15, 'start and end are from Version 15')
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 4, 3))], [make_node('Shape', ['x'], ['y'], start=1)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (2,))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])