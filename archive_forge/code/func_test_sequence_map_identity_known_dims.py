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
def test_sequence_map_identity_known_dims(self):
    input_value_infos = [make_tensor_value_info('input', TensorProto.FLOAT, (220, 220, 3))]
    output_value_infos = [make_tensor_value_info('output', TensorProto.FLOAT, (220, 220, 3))]
    body_graph = helper.make_graph([make_node('Identity', ['input'], ['output'])], 'body_graph', input_value_infos, output_value_infos)
    graph = self._make_graph([('input1', TensorProto.FLOAT, (220, 220, 3)), ('input2', TensorProto.FLOAT, (220, 220, 3)), ('input3', TensorProto.FLOAT, (220, 220, 3))], [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']), make_node('SequenceMap', ['in_sequence'], ['out_sequence'], body=body_graph)], [])
    self._assert_inferred(graph, [make_tensor_sequence_value_info('in_sequence', TensorProto.FLOAT, (220, 220, 3)), make_tensor_sequence_value_info('out_sequence', TensorProto.FLOAT, (220, 220, 3))])