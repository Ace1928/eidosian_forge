import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_upsample_6_7(self) -> None:
    from_opset = 6
    to_opset = 7
    data_type = TensorProto.FLOAT
    nodes = [onnx.helper.make_node('Upsample', inputs=['X'], outputs=['Y'], mode='nearest', width_scale=3.0, height_scale=2.0)]
    graph = helper.make_graph(nodes, 'test_upsample_6_7', [onnx.helper.make_tensor_value_info('X', data_type, [1, 1, 2, 2])], [onnx.helper.make_tensor_value_info('Y', data_type, [1, 1, 4, 6])])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert len(converted_model.graph.node) == 1
    assert converted_model.graph.node[0].op_type == 'Upsample'
    attribute_names = [attr.name for attr in converted_model.graph.node[0].attribute]
    assert 'scales' in attribute_names
    assert 'width_scale' not in attribute_names
    assert 'height_scale' not in attribute_names
    assert converted_model.opset_import[0].version == to_opset