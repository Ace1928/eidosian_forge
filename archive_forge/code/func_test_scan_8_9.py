import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_scan_8_9(self) -> None:
    from_opset = 8
    to_opset = 9
    data_type = TensorProto.FLOAT
    node1 = onnx.helper.make_node('Add', inputs=['sum_in', 'next'], outputs=['sum_out'])
    node2 = onnx.helper.make_node('Identity', inputs=['sum_out'], outputs=['scan_out'])
    g = onnx.helper.make_graph([node1, node2], 'scan_body', [onnx.helper.make_tensor_value_info('sum_in', data_type, [2]), onnx.helper.make_tensor_value_info('next', data_type, [2])], [onnx.helper.make_tensor_value_info('sum_out', data_type, [2]), onnx.helper.make_tensor_value_info('scan_out', data_type, [2])])
    no_sequence_lens = ''
    nodes = [onnx.helper.make_node('Scan', inputs=[no_sequence_lens, 'initial', 'x'], outputs=['y', 'z'], body=g, num_scan_inputs=1)]
    initial = onnx.helper.make_tensor_value_info('initial', data_type, [1, 2])
    x = onnx.helper.make_tensor_value_info('x', data_type, [1, 3, 2])
    y = onnx.helper.make_tensor_value_info('y', data_type, [1, 2])
    z = onnx.helper.make_tensor_value_info('z', data_type, [1, 3, 2])
    graph = onnx.helper.make_graph(nodes, 'test_scan_8_9', [initial, x], [y, z])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[0].op_type == 'Scan'
    assert converted_model.opset_import[0].version == to_opset