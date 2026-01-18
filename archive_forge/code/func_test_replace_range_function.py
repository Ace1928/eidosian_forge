import unittest
import numpy as np
from numpy.testing import assert_allclose
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnx.tools import update_model_dims
from onnx.tools.replace_constants import replace_initializer_by_constant_of_shape
def test_replace_range_function(self):
    dtype = np.float32
    value = np.random.randn(2, 100).astype(dtype)
    A = numpy_helper.from_array(value, name='A')
    value = np.array([1], dtype=dtype)
    C = numpy_helper.from_array(value, name='C')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    nodeC = helper.make_node('Constant', [], ['C'], value=C)
    node0 = helper.make_node('Constant', [], ['A'], value=A)
    node1 = helper.make_node('MatMul', ['X', 'A'], ['AX'])
    node2 = helper.make_node('Sub', ['AX', 'C'], ['Y'])
    opset_imports = [helper.make_opsetid('', onnx_opset_version()), helper.make_opsetid('custom', 1)]
    fct = helper.make_function('custom', 'unittest', ['X'], ['Y'], [nodeC, node0, node1, node2], opset_imports)
    node = helper.make_node('unittest', ['X'], ['Y'], domain='custom')
    graph = helper.make_graph([node], 'lr', [X], [Y], [C])
    model_def = helper.make_model(graph, functions=[fct], opset_imports=opset_imports)
    x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
    oinf1 = ReferenceEvaluator(model_def)
    y1 = oinf1.run(None, {'X': x})[0]
    repl = replace_initializer_by_constant_of_shape(model_def, use_range=True)
    node_types = {n.op_type for n in repl.functions[0].node}
    self.assertIn('Range', node_types)
    self.assertNotIn('ConstantOfShape', node_types)
    oinf2 = ReferenceEvaluator(repl)
    y2 = oinf2.run(None, {'X': x})[0]
    assert_allclose(y1.shape, y2.shape)