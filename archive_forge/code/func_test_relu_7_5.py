import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_relu_7_5(self) -> None:
    nodes = [helper.make_node('Relu', ['X'], ['Y'])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5,))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5,))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 7), 5)
    assert converted_model.graph.node[0].op_type == 'Relu'
    assert converted_model.opset_import[0].version == 5