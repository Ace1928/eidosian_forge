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
def test_roipool(self) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, (5, 3, 4, 4)), ('rois', TensorProto.INT64, (2, 5))], [make_node('MaxRoiPool', ['X', 'rois'], ['Y'], pooled_shape=[2, 2])], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (2, 3, 2, 2))])