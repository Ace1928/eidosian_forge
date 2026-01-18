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
def test_onehot_with_axis(self) -> None:
    graph = self._make_graph([('indices', TensorProto.INT64, (2, 3, 5)), ('depth', TensorProto.INT64, (1,)), ('values', TensorProto.FLOAT, (2,))], [make_node('OneHot', ['indices', 'depth', 'values'], 'Y', axis=1)], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (2, None, 3, 5))])