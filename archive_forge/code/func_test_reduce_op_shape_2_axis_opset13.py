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
def test_reduce_op_shape_2_axis_opset13(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (24, 4, 11))], [make_node('ReduceL1', 'x', 'y', axes=(1, 2), keepdims=0)], [], initializer=[make_tensor('axes', TensorProto.INT64, (2,), (1, 2))])
    operatorsetid = OperatorSetIdProto()
    operatorsetid.domain = ''
    operatorsetid.version = 13
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (24,))], opset_imports=[operatorsetid])