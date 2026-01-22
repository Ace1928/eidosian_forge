import os
import tempfile
import unittest
import numpy as np
import numpy.testing as npt
import onnx
import onnx.helper
import onnx.model_container
import onnx.numpy_helper
import onnx.reference
def _linear_regression():
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, None])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None])
    graph = onnx.helper.make_graph([onnx.helper.make_node('MatMul', ['X', 'A'], ['XA']), onnx.helper.make_node('MatMul', ['XA', 'B'], ['XB']), onnx.helper.make_node('MatMul', ['XB', 'C'], ['Y'])], 'mm', [X], [Y], [onnx.numpy_helper.from_array(np.arange(9).astype(np.float32).reshape((-1, 3)), name='A'), onnx.numpy_helper.from_array((np.arange(9) * 100).astype(np.float32).reshape((-1, 3)), name='B'), onnx.numpy_helper.from_array((np.arange(9) + 10).astype(np.float32).reshape((-1, 3)), name='C')])
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)
    return onnx_model