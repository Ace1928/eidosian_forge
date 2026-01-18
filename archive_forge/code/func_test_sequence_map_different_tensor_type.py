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
def test_sequence_map_different_tensor_type(self):
    body_graph = helper.make_graph(nodes=[make_node('Shape', ['x'], ['shape'])], name='body_graph', inputs=[onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, ('H', 'W', 'C'))], outputs=[onnx.helper.make_tensor_value_info('shape', onnx.TensorProto.INT64, (3,))])
    graph = self._make_graph([('input1', TensorProto.FLOAT, (220, 310, 3)), ('input2', TensorProto.FLOAT, (110, 210, 3)), ('input3', TensorProto.FLOAT, (90, 110, 3))], [make_node('SequenceConstruct', ['input1', 'input2', 'input3'], ['in_sequence']), make_node('SequenceMap', ['in_sequence'], ['shapes'], body=body_graph)], [])
    self._assert_inferred(graph, [make_tensor_sequence_value_info('in_sequence', TensorProto.FLOAT, (None, None, 3)), make_tensor_sequence_value_info('shapes', TensorProto.INT64, (3,))])