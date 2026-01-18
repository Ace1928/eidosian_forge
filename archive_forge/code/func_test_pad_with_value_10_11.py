import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_pad_with_value_10_11(self) -> None:
    pads = (0, 1, 2, 0, 2, 1)
    nodes = [helper.make_node('Pad', ['X'], ['Y'], pads=pads, value=1.0)]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (1, 2, 2))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (1, 5, 5))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 10), 11)
    assert converted_model.graph.node[1].op_type == 'Pad'
    assert converted_model.opset_import[0].version == 11