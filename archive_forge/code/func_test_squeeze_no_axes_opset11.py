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
def test_squeeze_no_axes_opset11(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (1, 3, 1, 1, 2, 1))], [make_node('Squeeze', ['x'], 'y')], [])
    operatorsetid = OperatorSetIdProto()
    operatorsetid.domain = ''
    operatorsetid.version = 11
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (3, 2))])