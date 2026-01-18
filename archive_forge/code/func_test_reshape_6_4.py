import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_reshape_6_4(self) -> None:
    nodes = [helper.make_node('Constant', [], ['shape'], value=helper.make_tensor('', TensorProto.INT64, [1], [5])), helper.make_node('Reshape', ['X', 'shape'], ['Y'])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5,))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5,))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 6), 4)
    assert converted_model.graph.node[0].op_type == 'Reshape'
    assert converted_model.opset_import[0].version == 4