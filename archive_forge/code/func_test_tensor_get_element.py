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
def test_tensor_get_element(self) -> None:
    tensor_type_proto = helper.make_tensor_type_proto(elem_type=TensorProto.DOUBLE, shape=[2, 1, 4])
    output_tensor_val_info = helper.make_value_info(name='output', type_proto=tensor_type_proto)
    graph = self._make_graph([('input', TensorProto.DOUBLE, (2, 1, 4))], [make_node('OptionalGetElement', ['input'], ['output'])], [])
    self._assert_inferred(graph, [output_tensor_val_info])