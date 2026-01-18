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
def test_optional_construct_sequence(self) -> None:
    tensor_type_proto = helper.make_tensor_type_proto(elem_type=TensorProto.INT64, shape=[2, 3, 0])
    sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
    sequence_val_info = helper.make_value_info(name='input_sequence', type_proto=sequence_type_proto)
    optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
    optional_val_info = helper.make_value_info(name='output_sequence', type_proto=optional_type_proto)
    graph = self._make_graph([('input1', TensorProto.INT64, (2, 3, 0))], [make_node('SequenceConstruct', ['input1'], ['input_sequence']), make_node('Optional', ['input_sequence'], ['output_sequence'])], [])
    self._assert_inferred(graph, [sequence_val_info, optional_val_info])