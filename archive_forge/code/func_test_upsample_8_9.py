import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_upsample_8_9(self) -> None:
    from_opset = 8
    to_opset = 9
    data_type = TensorProto.FLOAT
    nodes = [onnx.helper.make_node('Upsample', inputs=['X'], outputs=['Y'], mode='nearest', scales=[1.0, 1.0, 2.0, 3.0])]
    graph = helper.make_graph(nodes, 'test_upsample_8_9', [onnx.helper.make_tensor_value_info('X', data_type, [1, 1, 2, 2])], [onnx.helper.make_tensor_value_info('Y', data_type, [1, 1, 4, 6])])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert len(converted_model.graph.node) == 2
    assert converted_model.graph.node[0].op_type == 'Constant'
    assert converted_model.graph.node[1].op_type == 'Upsample'
    assert len(converted_model.graph.node[1].attribute) == 1
    assert converted_model.graph.node[1].attribute[0].name == 'mode'
    assert converted_model.opset_import[0].version == to_opset