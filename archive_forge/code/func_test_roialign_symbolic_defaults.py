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
def test_roialign_symbolic_defaults(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, ('N', 'C', 'H', 'W')), ('rois', TensorProto.FLOAT, ('num_rois', 4)), ('batch_indices', TensorProto.INT64, ('num_rois',))], [make_node('RoiAlign', ['x', 'rois', 'batch_indices'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ('num_rois', 'C', 1, 1))])