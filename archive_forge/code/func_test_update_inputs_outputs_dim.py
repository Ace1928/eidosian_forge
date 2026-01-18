import unittest
import numpy as np
from numpy.testing import assert_allclose
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnx.tools import update_model_dims
from onnx.tools.replace_constants import replace_initializer_by_constant_of_shape
def test_update_inputs_outputs_dim(self) -> None:
    node_def = helper.make_node('Conv', inputs=['x', 'W'], outputs=['y'], kernel_shape=[3, 3], strides=[2, 2])
    graph_def = helper.make_graph([node_def], 'test', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 5, 5]), helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 1, 3, 3])], [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2, 2])])
    model_def = helper.make_model(graph_def, producer_name='test')
    updated_def = update_model_dims.update_inputs_outputs_dims(model_def, {'x': [1, 1, 'x1', -1], 'W': [1, 1, 3, 3]}, {'y': [1, 1, -1, -1]})
    onnx.checker.check_model(updated_def)
    self.assertEqual(updated_def.graph.input[0].type.tensor_type.shape.dim[2].dim_param, 'x1')
    self.assertEqual(updated_def.graph.input[0].type.tensor_type.shape.dim[3].dim_param, 'x_3')
    self.assertEqual(updated_def.graph.output[0].type.tensor_type.shape.dim[2].dim_param, 'y_2')
    self.assertEqual(updated_def.graph.output[0].type.tensor_type.shape.dim[3].dim_param, 'y_3')