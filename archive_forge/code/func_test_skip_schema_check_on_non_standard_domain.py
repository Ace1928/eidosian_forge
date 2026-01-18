import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_skip_schema_check_on_non_standard_domain(self) -> None:
    node = helper.make_node('NonExistOp', ['X'], ['Y'], name='test', domain='test.domain')
    graph = helper.make_graph([node], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])])
    onnx_id = helper.make_opsetid('test.domain', 1)
    model = helper.make_model(graph, producer_name='test', opset_imports=[onnx_id])
    checker.check_model(model)