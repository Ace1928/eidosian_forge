import itertools
import math
import sys
import unittest
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from os import getenv
from textwrap import dedent
from typing import Sequence, Tuple
import numpy as np
import parameterized
import version_utils
from numpy.testing import assert_allclose
import onnx.reference.custom_element_types as custom
from onnx import (
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32, from_array
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun, OpRunExpand
from onnx.reference.ops import load_op
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
from onnx.reference.ops._op_list import Cast_19, Celu
from onnx.reference.ops.aionnx_preview_training._op_list import Adam
from onnx.reference.ops.op_celu import _vcelu1
from onnx.reference.ops.op_col2im import (
from onnx.reference.ops.op_conv import Conv, _conv_implementation
from onnx.reference.ops_optimized import Conv as ConvOptimized
from onnx.reference.ops_optimized.op_conv_optimized import _conv_implementation_im2col
class TestReferenceEvaluator(unittest.TestCase):
    m2_def = '\n        <\n            ir_version: 7,\n            opset_import: [ "": 10, "com.microsoft": 1]\n        >\n        agraph (float[N, M] B01, float[N, M] B11, float[N, M] B21) => (float[N, M] D0)\n        {\n            C0 = Add(B01, B11)\n            C1 = Sub(B11, B21)\n            D0 = Mul(C0, C1)\n        }\n        '

    @staticmethod
    def _load_model(m_def: str) -> ModelProto:
        """Parses a model from a string representation, including checking
        the model for correctness
        """
        m = parser.parse_model(m_def)
        checker.check_model(m)
        return m

    @staticmethod
    def _linear_regression(clip=False, opset=None, min_value=-1.0, max_value=1.0):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        if clip:
            node2 = make_node('Add', ['XA', 'B'], ['Y_clip'])
            if opset is not None and opset < 11:
                if min_value:
                    if max_value:
                        node3 = make_node('Clip', ['Y_clip'], ['Y'], min=min_value, max=max_value)
                    else:
                        node3 = make_node('Clip', ['Y_clip'], ['Y'], min=min_value)
                elif max_value:
                    node3 = make_node('Clip', ['Y_clip'], ['Y'], max=max_value)
                else:
                    node3 = make_node('Clip', ['Y_clip'], ['Y'])
                graph = make_graph([node1, node2, node3], 'lr', [X, A, B], [Y])
            else:
                mi = from_array(np.array([min_value], dtype=np.float32), name='mi') if min_value else None
                ma = from_array(np.array([max_value], dtype=np.float32), name='ma') if max_value else None
                inputs = ['Y_clip', 'mi' if mi else '', 'ma' if ma else '']
                node3 = make_node('Clip', inputs, ['Y'])
                initializer = [_ for _ in [mi, ma] if _]
                graph = make_graph([node1, node2, node3], 'lr', [X, A, B], [Y], initializer=initializer)
            f = lambda x, a, b: np.clip(a @ a + b, min_value, max_value)
        else:
            node2 = make_node('Add', ['XA', 'B'], ['Y'])
            graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
            f = lambda x, a, b: a @ a + b
        if opset is None:
            onnx_model = make_model(graph)
        else:
            onnx_model = make_model(graph, opset_imports=[make_opsetid('', opset)])
        try:
            check_model(onnx_model)
        except Exception as e:
            raise AssertionError(f'checker fails for\n{onnx_model}') from e
        return (onnx_model, f)

    def test_reference_evaluator_exceptions(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        with self.assertRaises(TypeError):
            ReferenceEvaluator(X)

    def test_reference_evaluator_no_attribute(self):
        m = TestReferenceEvaluator._load_model(TestReferenceEvaluator.m2_def)
        checker.check_model(m)
        sess = ReferenceEvaluator(m)
        self.assertEqual(sess.input_names, ['B01', 'B11', 'B21'])
        self.assertEqual(sess.output_names, ['D0'])
        self.assertEqual(sess.opsets, {'': 10, 'com.microsoft': 1})
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)
        res = sess.run(None, {'B01': x, 'B11': y, 'B21': z})[0]
        expected = (x + y) * (y - z)
        assert_allclose(expected, res)

    def test_reference_evaluator_no_attribute_bytes(self):
        m = TestReferenceEvaluator._load_model(TestReferenceEvaluator.m2_def)
        checker.check_model(m)
        sess = ReferenceEvaluator(m.SerializeToString())
        self.assertEqual(sess.input_names, ['B01', 'B11', 'B21'])
        self.assertEqual(sess.output_names, ['D0'])
        self.assertEqual(sess.opsets, {'': 10, 'com.microsoft': 1})
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)
        res = sess.run(None, {'B01': x, 'B11': y, 'B21': z})[0]
        expected = (x + y) * (y - z)
        assert_allclose(expected, res)

    def test_reference_evaluator_no_attribute_verbose(self):
        m = TestReferenceEvaluator._load_model(TestReferenceEvaluator.m2_def)
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        y = np.array([[4, 5], [6, 7]], dtype=np.float32)
        z = np.array([[-4, -5], [-6, -7]], dtype=np.float32)
        with self.subTest(level=2):
            sess = ReferenceEvaluator(m, verbose=2)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {'B01': x, 'B11': y, 'B21': z})
            out = stdout.getvalue()
            log = 'Add(B01, B11) -> C0\nSub(B11, B21) -> C1\nMul(C0, C1) -> D0\n'
            self.assertEqual(log, out)
        with self.subTest(level=3):
            sess = ReferenceEvaluator(m, verbose=3)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {'B01': x, 'B11': y, 'B21': z})
            out = stdout.getvalue()
            log = dedent('\n                 +I B01: float32:(2, 2) in [0.0, 3.0]\n                 +I B11: float32:(2, 2) in [4.0, 7.0]\n                 +I B21: float32:(2, 2) in [-7.0, -4.0]\n                Add(B01, B11) -> C0\n                 + C0: float32:(2, 2) in [4.0, 10.0]\n                Sub(B11, B21) -> C1\n                 + C1: float32:(2, 2) in [8.0, 14.0]\n                Mul(C0, C1) -> D0\n                 + D0: float32:(2, 2) in [32.0, 140.0]\n                ').lstrip('\n')
            self.assertEqual(log, out)
        with self.subTest(level=4):
            sess = ReferenceEvaluator(m, verbose=4)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {'B01': x, 'B11': y, 'B21': z})
            out = stdout.getvalue()
            log = dedent('\n                 +I B01: float32:(2, 2):[0.0, 1.0, 2.0, 3.0]\n                 +I B11: float32:(2, 2):[4.0, 5.0, 6.0, 7.0]\n                 +I B21: float32:(2, 2):[-4.0, -5.0, -6.0, -7.0]\n                Add(B01, B11) -> C0\n                 + C0: float32:(2, 2):[4.0, 6.0, 8.0, 10.0]\n                Sub(B11, B21) -> C1\n                 + C1: float32:(2, 2):[8.0, 10.0, 12.0, 14.0]\n                Mul(C0, C1) -> D0\n                 + D0: float32:(2, 2):[32.0, 60.0, 96.0, 140.0]\n                ').lstrip('\n')
            self.assertEqual(log, out)
        with self.subTest(level=15):
            sess = ReferenceEvaluator(m, verbose=15)
            stdout = StringIO()
            with redirect_stdout(stdout):
                sess.run(None, {'B01': x, 'B11': y, 'B21': z})
            out = stdout.getvalue()
            log = dedent('\n                 +I B01: float32:(2, 2):[0.0, 1.0, 2.0, 3.0]\n                 +I B11: float32:(2, 2):[4.0, 5.0, 6.0, 7.0]\n                 +I B21: float32:(2, 2):[-4.0, -5.0, -6.0, -7.0]\n                Add(B01, B11) -> C0\n                -- begin Add.run(2 inputs)\n                -- done Add.run -> 1 outputs\n                 + C0: float32:(2, 2):[4.0, 6.0, 8.0, 10.0]\n                Sub(B11, B21) -> C1\n                -- begin Sub.run(2 inputs)\n                -- done Sub.run -> 1 outputs\n                 + C1: float32:(2, 2):[8.0, 10.0, 12.0, 14.0]\n                Mul(C0, C1) -> D0\n                -- begin Mul.run(2 inputs)\n                -- done Mul.run -> 1 outputs\n                 + D0: float32:(2, 2):[32.0, 60.0, 96.0, 140.0]\n                ').lstrip('\n')
            self.assertEqual(log, out)

    def test_reference_evaluator_lr(self):
        lr, f = TestReferenceEvaluator._linear_regression()
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        a = np.array([1, 1], dtype=np.float32)
        b = np.array([11], dtype=np.float32)
        expected = f(x, a, b)
        sess = ReferenceEvaluator(lr)
        got = sess.run(None, {'X': a, 'A': a, 'B': b})[0]
        assert_allclose(expected, got)

    def test_reference_evaluator_lr_clip(self):
        with self.subTest(opt='min+max'):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, 'Clip_11')
            got = sess.run(None, {'X': a, 'A': a, 'B': b})[0]
            assert_allclose(expected, got)
        with self.subTest(opt='max'):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True, min_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, 'Clip_11')
            got = sess.run(None, {'X': a, 'A': a, 'B': b})[0]
            assert_allclose(expected, got)
        with self.subTest(opt='min'):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True, max_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, 'Clip_11')
            got = sess.run(None, {'X': a, 'A': a, 'B': b})[0]
            assert_allclose(expected, got)

    def test_reference_evaluator_lr_clip_6(self):
        with self.subTest(opt='min+max'):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True, opset=10)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, 'Clip_6')
            self.assertEqual(last_node.min, -1)
            self.assertEqual(last_node.max, 1)
            got = sess.run(None, {'X': a, 'A': a, 'B': b})[0]
            assert_allclose(expected, got)
        with self.subTest(opt='max'):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True, opset=10, min_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, 'Clip_6')
            self.assertEqual(last_node.max, 1)
            self.assertEqual(last_node.min, -3.4028234663852886e+38)
            got = sess.run(None, {'X': a, 'A': a, 'B': b})[0]
            assert_allclose(expected, got)
        with self.subTest(opt='min'):
            lr, f = TestReferenceEvaluator._linear_regression(clip=True, opset=10, max_value=None)
            x = np.array([[0, 1], [2, 3]], dtype=np.float32)
            a = np.array([1, 1], dtype=np.float32)
            b = np.array([11], dtype=np.float32)
            expected = f(x, a, b)
            sess = ReferenceEvaluator(lr)
            last_node = sess.rt_nodes_[-1]
            self.assertEqual(last_node.__class__.__name__, 'Clip_6')
            self.assertEqual(last_node.min, -1)
            self.assertEqual(last_node.max, 3.4028234663852886e+38)
            got = sess.run(None, {'X': a, 'A': a, 'B': b})[0]
            assert_allclose(expected, got)

    def test_nested_local_functions(self):
        m = parser.parse_model('\n            <\n              ir_version: 8,\n              opset_import: [ "" : 14, "local" : 1],\n              producer_name: "test",\n              producer_version: "1.0",\n              model_version: 1,\n              doc_string: "Test preprocessing model"\n            >\n            agraph (uint8[H, W, C] x) => (uint8[H, W, C] x_processed)\n            {\n                x_processed = local.func(x)\n            }\n\n            <\n              opset_import: [ "" : 14 ],\n              domain: "local",\n              doc_string: "function 1"\n            >\n            f1 (x) => (y) {\n                y = Identity(x)\n            }\n\n            <\n              opset_import: [ "" : 14 ],\n              domain: "local",\n              doc_string: "function 2"\n            >\n            f2 (x) => (y) {\n                y = Identity(x)\n            }\n\n            <\n              opset_import: [ "" : 14, "local" : 1 ],\n              domain: "local",\n              doc_string: "Preprocessing function."\n            >\n            func (x) => (y) {\n                x1 = local.f1(x)\n                y = local.f2(x1)\n            }\n        ')
        sess = ReferenceEvaluator(m)
        x = np.array([0, 1, 3], dtype=np.uint8).reshape((1, 1, 3))
        result = sess.run(None, {'x': x})[0]
        expected = x
        assert_allclose(expected, result)

    def test_reduce_sum_11(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('ReduceSum', ['X'], ['Y'], axes=[1], keepdims=1)
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 11)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        expected = x.sum(axis=1, keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_square_11(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('ReduceSumSquare', ['X'], ['Y'], axes=[1], keepdims=1)
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 11)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        expected = (x * x).sum(axis=1, keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_13(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.INT64, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('ReduceSum', ['X', 'A'], ['Y'], keepdims=1)
        graph = make_graph([node1], 'rs', [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([1], dtype=np.int64)
        expected = x.sum(axis=1, keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x, 'A': a})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_attribute(self):
        opset = onnx_opset_version()
        new_domain = 'custom'
        opset_imports = [make_opsetid('', opset), make_opsetid(new_domain, 1)]
        node = make_node('ReduceSum', ['X', 'axis'], ['Y'])
        att = AttributeProto()
        att.name = 'keepdims'
        att.ref_attr_name = 'keepdims'
        att.type = AttributeProto.INT
        node.attribute.append(att)
        my_reduce_sum = make_function(new_domain, 'MyReduceSum', ['X', 'axis'], ['Y'], [node], opset_imports, ['keepdims'])
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        axis = make_tensor_value_info('axis', TensorProto.INT64, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        graph = make_graph([make_node('MyReduceSum', ['X', 'axis'], ['Y'], domain=new_domain, keepdims=1)], 'example', [X, axis], [Y])
        onnx_model = make_model(graph, opset_imports=opset_imports, functions=[my_reduce_sum])
        sess = ReferenceEvaluator(onnx_model)
        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([-1], dtype=np.int64)
        result = sess.run(None, {'X': x, 'axis': a})[0]
        expected = x.sum(axis=-1, keepdims=1)
        assert_allclose(expected, result)

    def test_reduce_sum_square_18(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.INT64, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('ReduceSumSquare', ['X', 'A'], ['Y'], keepdims=1)
        graph = make_graph([node1], 'rs', [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([1], dtype=np.int64)
        expected = (x * x).sum(axis=1, keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x, 'A': a})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_13_empty_axes(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.INT64, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('ReduceSum', ['X', 'A'], ['Y'], keepdims=1)
        graph = make_graph([node1], 'rs', [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([], dtype=np.int64)
        expected = x.sum(keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x, 'A': a})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_square_18_empty_axes(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.INT64, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('ReduceSumSquare', ['X', 'A'], ['Y'], keepdims=1)
        graph = make_graph([node1], 'rs', [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        a = np.array([], dtype=np.int64)
        expected = (x * x).sum(keepdims=1)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x, 'A': a})[0]
        assert_allclose(expected, got)

    def test_reduce_sum_13_empty_axes_noop(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('ReduceSum', ['X'], ['Y'], keepdims=1, noop_with_empty_axes=1)
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 13)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x})[0]
        assert_allclose(x, got)

    def test_reduce_sum_square_18_empty_axes_noop(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('ReduceSumSquare', ['X'], ['Y'], keepdims=1, noop_with_empty_axes=1)
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        check_model(onnx_model)
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x})[0]
        assert_allclose(x * x, got)

    def test_greater(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None])
        node1 = make_node('Greater', ['X', 'Y'], ['Z'])
        graph = make_graph([node1], 'g', [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 13)])
        check_model(onnx_model)
        x = np.arange(4).reshape((2, 2)).astype(np.float32)
        y = np.array([2], dtype=np.float32)
        expected = x > y
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x, 'Y': y})[0]
        assert_allclose(expected, got)

    def test_node_proto(self):
        node1 = make_node('Greater', ['X', 'Y'], ['Z'])
        x = np.arange(4).reshape((2, 2)).astype(np.float32)
        y = np.array([2], dtype=np.float32)
        expected = x > y
        sess = ReferenceEvaluator(node1)
        got = sess.run(None, {'X': x, 'Y': y})[0]
        assert_allclose(expected, got)

    def test_greater_or_equal(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None])
        node1 = make_node('GreaterOrEqual', ['X', 'Y'], ['Z'])
        graph = make_graph([node1], 'g', [X, Y], [Z])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 13)])
        check_model(onnx_model)
        x = np.arange(4).reshape((2, 2)).astype(np.float32)
        y = np.array([2], dtype=np.float32)
        expected = x >= y
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': x, 'Y': y})[0]
        assert_allclose(expected, got)

    def test_if(self):
        C = make_tensor_value_info('C', TensorProto.FLOAT, [None])
        bthen = make_node('Constant', [], ['C'], value_floats=from_array(np.array([1], dtype=np.float32)))
        bthen_body = make_graph([bthen], 'gthen', [], [C])
        C = make_tensor_value_info('C', TensorProto.FLOAT, [None])
        belse = make_node('Constant', [], ['C'], value_floats=from_array(np.array([0], dtype=np.float32)))
        belse_body = make_graph([belse], 'gelse', [], [C])
        zero = from_array(np.array([0], dtype=np.float32), name='zero')
        greater = make_node('Greater', ['X', 'zero'], ['G'])
        node_if = make_node('If', ['G'], ['Z'], then_branch=bthen_body, else_branch=belse_body)
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Z = make_tensor_value_info('Z', TensorProto.FLOAT, [None])
        graph = make_graph([greater, node_if], 'g', [X], [Z], initializer=[zero])
        model_def = make_model(graph)
        sess = ReferenceEvaluator(model_def)
        self.assertEqual(str(sess), 'ReferenceEvaluator(X) -> Z')
        x = np.array([1], dtype=np.float32)
        got = sess.run(None, {'X': x})[0]
        assert_allclose(np.array([1], dtype=np.float32), got)
        x = np.array([-1], dtype=np.float32)
        got = sess.run(None, {'X': x})[0]
        assert_allclose(np.array([0], dtype=np.float32), got)

    def test_if_function(self):
        then_out = make_tensor_value_info('then_out', TensorProto.FLOAT, [5])
        else_out = make_tensor_value_info('else_out', TensorProto.FLOAT, [5])
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        y = np.array([5, 4, 3, 2, 1]).astype(np.float32)
        then_const_node = make_node('Constant', inputs=[], outputs=['then_out'], value=from_array(x))
        else_const_node = make_node('Constant', inputs=[], outputs=['else_out'], value=from_array(y))
        then_body = make_graph([then_const_node], 'then_body', [], [then_out])
        else_body = make_graph([else_const_node], 'else_body', [], [else_out])
        if_node = make_node('If', inputs=['f_cond'], outputs=['f_res'], then_branch=then_body, else_branch=else_body)
        f = FunctionProto()
        f.domain = 'custom'
        f.name = 'fn'
        f.input.extend(['f_cond'])
        f.output.extend(['f_res'])
        f.node.extend([if_node])
        opset = onnx_opset_version()
        f.opset_import.extend([make_opsetid('', opset)])
        graph = make_graph(nodes=[make_node('fn', domain='custom', inputs=['cond'], outputs=['res'])], name='graph', inputs=[make_tensor_value_info('cond', TensorProto.BOOL, [])], outputs=[make_tensor_value_info('res', TensorProto.FLOAT, [5])])
        m = make_model(graph, producer_name='test', opset_imports=[make_opsetid('', opset), make_opsetid('custom', 1)])
        m.functions.extend([f])
        sess = ReferenceEvaluator(m)
        result = sess.run(None, {'cond': np.array(True)})
        expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        assert_allclose(expected, result[0])

    def test_function_attribute(self):
        opset = onnx_opset_version()
        new_domain = 'custom'
        opset_imports = [make_opsetid('', opset), make_opsetid(new_domain, 1)]
        cst = make_node('Constant', [], ['B'])
        att = AttributeProto()
        att.name = 'value'
        att.ref_attr_name = 'bias'
        att.type = AttributeProto.TENSOR
        cst.attribute.append(att)
        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])
        linear_regression = make_function(new_domain, 'LinearRegression', ['X', 'A'], ['Y'], [cst, node1, node2], opset_imports, ['bias'])
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        graph = make_graph([make_node('LinearRegression', ['X', 'A'], ['Y1'], domain=new_domain, bias=make_tensor('former_B', TensorProto.FLOAT, [1], [0.67])), make_node('Abs', ['Y1'], ['Y'])], 'example', [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=opset_imports, functions=[linear_regression])
        sess = ReferenceEvaluator(onnx_model)
        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([1, -1], dtype=np.float32)
        result = sess.run(None, {'X': x, 'A': a})[0]
        expected = np.abs(x @ a + 0.67)
        assert_allclose(expected, result)

    def test_function_attribute_nested_graph(self):
        opset = onnx_opset_version()
        new_domain = 'custom'
        opset_imports = [make_opsetid('', opset), make_opsetid(new_domain, 1)]
        cst1 = make_node('Constant', [], ['B1'])
        att = AttributeProto()
        att.name = 'value'
        att.ref_attr_name = 'bias1'
        att.type = AttributeProto.TENSOR
        cst1.attribute.append(att)
        cst2 = make_node('Constant', [], ['B2'])
        att = AttributeProto()
        att.name = 'value'
        att.ref_attr_name = 'bias2'
        att.type = AttributeProto.TENSOR
        cst2.attribute.append(att)
        then_out = make_tensor_value_info('B1', TensorProto.FLOAT, [None])
        else_out = make_tensor_value_info('B2', TensorProto.FLOAT, [None])
        then_body = make_graph([cst1], 'then_body', [], [then_out])
        else_body = make_graph([cst2], 'else_body', [], [else_out])
        zero = make_node('Constant', inputs=[], outputs=['zero'], value=from_array(np.array([0], dtype=np.float32)))
        mini = make_node('ReduceMin', ['X'], ['Xmin'])
        f_cond = make_node('Greater', ['Xmin', 'zero'], ['f_cond'])
        if_node = make_node('If', inputs=['f_cond'], outputs=['B'], then_branch=then_body, else_branch=else_body)
        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])
        linear_regression = make_function(new_domain, 'LinearRegression', ['X', 'A'], ['Y'], [zero, mini, f_cond, if_node, node1, node2], opset_imports, ['bias1', 'bias2'])
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        graph = make_graph([make_node('LinearRegression', ['X', 'A'], ['Y1'], domain=new_domain, bias1=make_tensor('former_B1', TensorProto.FLOAT, [1], [0.67]), bias2=make_tensor('former_B2', TensorProto.FLOAT, [1], [777])), make_node('Abs', ['Y1'], ['Y'])], 'example', [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=opset_imports, functions=[linear_regression])
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        self.assertEqual(sess.rt_nodes_[0].__class__.__name__, 'OpFunction')
        self.assertEqual(sess.rt_nodes_[0].impl_.__class__.__name__, 'ReferenceEvaluator')
        fct = sess.rt_nodes_[0].impl_
        checked = False
        for node in fct.rt_nodes_:
            if node.__class__.__name__.startswith('If'):
                if not node.has_linked_attribute:
                    raise AssertionError(f'Nested node {type(node)} declares no linked attribute but a subgraph does.')
                checked = True
        if not checked:
            raise AssertionError("No node 'If' was found, has_linked_attribute could not be checked.")
        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([1, -1], dtype=np.float32)
        result = sess.run(None, {'X': x + 1, 'A': a})[0]
        expected = np.abs(x @ a + 0.67)
        assert_allclose(expected, result)
        result = sess.run(None, {'X': x - 10, 'A': a})[0]
        expected = np.abs(x @ a + 777)
        assert_allclose(expected, result)

    def test_function_attribute_nested_nested_graph(self):
        opset = onnx_opset_version()
        new_domain = 'custom'
        opset_imports = [make_opsetid('', opset), make_opsetid(new_domain, 1)]
        cst1 = make_node('Constant', [], ['B1'])
        att = AttributeProto()
        att.name = 'value'
        att.ref_attr_name = 'bias1'
        att.type = AttributeProto.TENSOR
        cst1.attribute.append(att)
        cst2 = make_node('Constant', [], ['B2'])
        att = AttributeProto()
        att.name = 'value'
        att.ref_attr_name = 'bias2'
        att.type = AttributeProto.TENSOR
        cst2.attribute.append(att)
        then_out = make_tensor_value_info('B1', TensorProto.FLOAT, [None])
        else_out = make_tensor_value_info('B2', TensorProto.FLOAT, [None])
        then_body1 = make_graph([cst1], 'then_body', [], [then_out])
        else_body1 = make_graph([cst2], 'else_body', [], [else_out])
        c100 = make_node('Constant', inputs=[], outputs=['c100'], value=from_array(np.array([100], dtype=np.float32)))
        f_cond = make_node('Greater', ['Xmin', 'c100'], ['f_cond_100'])
        if_node = make_node('If', inputs=['f_cond_100'], outputs=['B4'], then_branch=then_body1, else_branch=else_body1)
        cst3 = make_node('Constant', [], ['B3'])
        att = AttributeProto()
        att.name = 'value'
        att.ref_attr_name = 'bias3'
        att.type = AttributeProto.TENSOR
        cst3.attribute.append(att)
        then_out = make_tensor_value_info('B3', TensorProto.FLOAT, [None])
        then_body2 = make_graph([cst3], 'then_body', [], [then_out])
        else_out = make_tensor_value_info('B4', TensorProto.FLOAT, [None])
        else_body2 = make_graph([c100, f_cond, if_node], 'else_body', [], [else_out])
        zero = make_node('Constant', inputs=[], outputs=['zero'], value=from_array(np.array([0], dtype=np.float32)))
        mini = make_node('ReduceMin', ['X'], ['Xmin'])
        f_cond = make_node('Less', ['Xmin', 'zero'], ['f_cond_zero'])
        if_node = make_node('If', inputs=['f_cond_zero'], outputs=['B'], then_branch=then_body2, else_branch=else_body2)
        node1 = make_node('MatMul', ['X', 'A'], ['XA'])
        node2 = make_node('Add', ['XA', 'B'], ['Y'])
        linear_regression = make_function(new_domain, 'LinearRegression', ['X', 'A'], ['Y'], [zero, mini, f_cond, if_node, node1, node2], opset_imports, ['bias1', 'bias2', 'bias3'])
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        graph = make_graph([make_node('LinearRegression', ['X', 'A'], ['Y1'], domain=new_domain, bias1=make_tensor('former_B1', TensorProto.FLOAT, [1], [0.67]), bias2=make_tensor('former_B2', TensorProto.FLOAT, [1], [777]), bias3=make_tensor('former_B3', TensorProto.FLOAT, [1], [-888])), make_node('Abs', ['Y1'], ['Y'])], 'example', [X, A], [Y])
        onnx_model = make_model(graph, opset_imports=opset_imports, functions=[linear_regression])
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        x = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = np.array([1, -1], dtype=np.float32)
        result = sess.run(None, {'X': x + 1, 'A': a})[0]
        expected = np.abs(x @ a + 777)
        assert_allclose(expected, result)
        result = sess.run(None, {'X': x - 10, 'A': a})[0]
        expected = np.abs(x @ a - 888)
        assert_allclose(expected, result)
        result = sess.run(None, {'X': x + 1000, 'A': a})[0]
        expected = np.abs(x @ a + 0.67)
        assert_allclose(expected, result)

    def test_custom_node(self):

        class _InvAlpha:
            op_domain = 'custom'

            def __init__(self, onnx_node, run_params):
                self.onnx_node = onnx_node
                self.run_params = run_params

            def _run(self, x):
                return (1 / (x + self.alpha),)

        class InvAlpha2(OpRun):

            def _run(self, x):
                return (1 / (x + self.alpha),)

        class InvAlpha(OpRun):
            op_domain = 'custom'

            def _run(self, x, alpha=None):
                alpha = alpha or self.alpha
                return (1 / (x + alpha),)
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        with self.assertRaises(NotImplementedError):
            ReferenceEvaluator(onnx_model)
        node1 = make_node('_InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
        with self.assertRaises(TypeError):
            ReferenceEvaluator(onnx_model, new_ops=[_InvAlpha])
        node1 = make_node('InvAlpha2', ['X'], ['Y'], alpha=0.5, domain='custom')
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
        with self.assertRaises(NotImplementedError):
            ReferenceEvaluator(onnx_model, new_ops=[InvAlpha2])
        node1 = make_node('InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
        sess = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha, InvAlpha])
        got = sess.run(None, {'X': x})[0]
        expected = 1 / (x + 0.5)
        assert_allclose(expected, got)

    def test_custom_no_output_tuple(self):

        class InvAlpha(OpRun):
            op_domain = 'custom'

            def _run(self, x, alpha=None):
                alpha = alpha or self.alpha
                return 1 / (x + alpha)
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        ref = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha])
        with self.assertRaises(TypeError):
            ref.run(None, {'X': x})

    def test_custom_empty_output(self):

        class InvAlpha(OpRun):
            op_domain = 'custom'

            def _run(self, x, alpha=None):
                return tuple()
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        ref = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha])
        with self.assertRaises(ValueError):
            ref.run(None, {'X': x})

    def test_custom_tuple_tuple(self):

        class InvAlpha(OpRun):
            op_domain = 'custom'

            def _run(self, x, alpha=None):
                alpha = alpha or self.alpha
                res = tuple([tuple([1 / (x + alpha)])])
                assert isinstance(res, tuple)
                assert isinstance(res[0], tuple)
                return res
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        ref = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha])
        with self.assertRaises(TypeError):
            ref.run(None, {'X': x})

    def test_custom_tuple_unexpected_type(self):

        class CustomType:
            pass

        class InvAlpha(OpRun):
            op_domain = 'custom'

            def _run(self, x, alpha=None):
                res = tuple([CustomType()])
                assert isinstance(res, tuple)
                assert isinstance(res[0], CustomType)
                return res
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
        graph = make_graph([node1], 'rs', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
        x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
        ref = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha])
        with self.assertRaises(TypeError):
            ref.run(None, {'X': x})

    def test_loop(self):
        cond_in = make_tensor_value_info('cond_in', TensorProto.BOOL, [])
        cond_out = make_tensor_value_info('cond_out', TensorProto.BOOL, [])
        iter_count = make_tensor_value_info('iter_count', TensorProto.INT64, [])
        seq_in = make_tensor_sequence_value_info('seq_in', TensorProto.FLOAT, None)
        seq_out = make_tensor_sequence_value_info('seq_out', TensorProto.FLOAT, None)
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        x_const_node = make_node('Constant', inputs=[], outputs=['x'], value=make_tensor(name='const_tensor_x', data_type=TensorProto.FLOAT, dims=x.shape, vals=x.flatten().astype(float)))
        one_const_node = make_node('Constant', inputs=[], outputs=['one'], value=make_tensor(name='const_tensor_one', data_type=TensorProto.INT64, dims=(), vals=[1]))
        zero_const_node = make_node('Constant', inputs=[], outputs=['slice_start'], value=make_tensor(name='const_tensor_zero', data_type=TensorProto.INT64, dims=(1,), vals=[0]))
        axes_node = make_node('Constant', inputs=[], outputs=['axes'], value=make_tensor(name='const_tensor_axes', data_type=TensorProto.INT64, dims=(), vals=[0]))
        add_node = make_node('Add', inputs=['iter_count', 'one'], outputs=['end'])
        end_unsqueeze_node = make_node('Unsqueeze', inputs=['end', 'axes'], outputs=['slice_end'])
        slice_node = make_node('Slice', inputs=['x', 'slice_start', 'slice_end'], outputs=['slice_out'])
        insert_node = make_node('SequenceInsert', inputs=['seq_in', 'slice_out'], outputs=['seq_out'])
        identity_node = make_node('Identity', inputs=['cond_in'], outputs=['cond_out'])
        loop_body = make_graph([identity_node, x_const_node, one_const_node, zero_const_node, add_node, axes_node, end_unsqueeze_node, slice_node, insert_node], 'loop_body', [iter_count, cond_in, seq_in], [cond_out, seq_out])
        node = make_node('Loop', inputs=['trip_count', 'cond', 'seq_empty'], outputs=['seq_res'], body=loop_body)
        node_concat = make_node('ConcatFromSequence', inputs=['seq_res'], outputs=['res'], axis=0, new_axis=0)
        trip_count = np.array(5).astype(np.int64)
        seq_empty = []
        cond = np.array(1).astype(np.bool_)
        model_def = make_model(graph=make_graph(name='loop_test', inputs=[make_tensor_value_info('trip_count', TensorProto.INT64, trip_count.shape), make_tensor_value_info('cond', TensorProto.BOOL, cond.shape), make_sequence_value_info('seq_empty', TensorProto.FLOAT, [])], outputs=[make_tensor_value_info('res', TensorProto.FLOAT, None)], nodes=[node, node_concat]))
        expected = np.array([1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        oinf = ReferenceEvaluator(model_def)
        inputs = {'trip_count': trip_count, 'cond': cond, 'seq_empty': seq_empty}
        got = oinf.run(None, inputs)
        assert_allclose(expected, got[0])

    def test_onnxt_runtime_bernoulli(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('Bernoulli', ['X'], ['Y'], seed=0.0)
        graph = make_graph([node1], 'g', [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': np.zeros((2, 4), dtype=np.float32) + 0.5})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)
        self.assertGreater(got.min(), -1e-05)
        self.assertLess(got.max(), 1 + 1e-05)

    def test_onnxt_runtime_random_uniform(self):
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('RandomUniform', [], ['Y'], seed=0.0, shape=[2, 4])
        graph = make_graph([node1], 'g', [], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)
        self.assertGreater(got.min(), 0)
        self.assertLess(got.max(), 1)

    def test_onnxt_runtime_random_uniform_like(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('RandomUniformLike', ['X'], ['Y'], seed=0.0)
        graph = make_graph([node1], 'g', [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': np.zeros((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)
        self.assertGreater(got.min(), 0)
        self.assertLess(got.max(), 1)

    def test_onnxt_runtime_random_normal(self):
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('RandomNormal', [], ['Y'], seed=0.0, shape=[2, 4])
        graph = make_graph([node1], 'g', [], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)

    def test_onnxt_runtime_random_normal_like(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        node1 = make_node('RandomNormalLike', ['X'], ['Y'], seed=0.0)
        graph = make_graph([node1], 'g', [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        got = sess.run(None, {'X': np.zeros((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (2, 4))
        self.assertEqual(got.dtype, np.float32)

    def test_eval_celu(self):
        inst = Celu.create(alpha=0.5)
        self.assertEqual(inst.alpha, 0.5)
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = Celu.eval(x, alpha=0.5)
        expected = _vcelu1(x, alpha=0.5)
        assert_allclose(expected, y)

    def test_eval_cast(self):
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = Cast_19.eval(x, to=TensorProto.FLOAT8E4M3FN)
        dy = Cast_19.eval(y, to=TensorProto.FLOAT)
        expected = x
        assert_allclose(expected, dy)

    def test_eval_celu_load_op(self):
        celu = load_op('', 'Celu')
        self.assertEqual(celu.op_domain, '')
        inst = celu.create(alpha=0.5)
        self.assertEqual(inst.alpha, 0.5)
        x = np.array([[0, 1], [-1, 2]], dtype=np.float32)
        y = celu.eval(x, alpha=0.5)
        expected = _vcelu1(x, alpha=0.5)
        assert_allclose(expected, y)

    def test_create_adam(self):
        inst = Adam.create(alpha=0.5)
        self.assertEqual(inst.alpha, 0.5)

    @skip_if_no_onnxruntime
    def test_conv(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info('W', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('Conv', ['X', 'W', 'B'], ['Y'], pads=[1, 1, 1, 1], dilations=[1, 1], strides=[2, 2])
        graph = make_graph([node], 'g', [X, W, B], [Y])
        onnx_model = make_model_gen_version(graph, opset_imports=[make_opsetid('', 16)])
        sess1 = run_ort_inference(onnx_model)
        if sess1 is None:
            return
        sess2 = ReferenceEvaluator(onnx_model, optimized=False)
        self.assertIsInstance(sess2.rt_nodes_[0], Conv)
        sess3 = ReferenceEvaluator(onnx_model, new_ops=[ConvOptimized], optimized=False)
        self.assertIsInstance(sess3.rt_nodes_[0], ConvOptimized)
        sess4 = ReferenceEvaluator(onnx_model, optimized=True)
        self.assertIsInstance(sess4.rt_nodes_[0], ConvOptimized)
        sH, sW = (5, 6)
        for i in range(sH):
            for j in range(sW):
                X = np.zeros((1, 1, sH, sW), dtype=np.float32)
                X[0, 0, i, j] = 1.0
                W = np.zeros((1, 1, 3, 3), dtype=np.float32)
                W[0, 0, :, :] = np.minimum(2 ** np.arange(9).reshape((3, -1)), 256)
                B = np.array([[[[0]]]], dtype=np.float32)
                expected = sess1.run(None, {'X': X, 'W': W, 'B': B})[0]
                got = sess2.run(None, {'X': X, 'W': W, 'B': B})[0]
                assert_allclose(expected, got)
                got3 = sess3.run(None, {'X': X, 'W': W, 'B': B})[0]
                assert_allclose(expected, got3)
                got4 = sess4.run(None, {'X': X, 'W': W, 'B': B})[0]
                assert_allclose(expected, got4)

    @skip_if_no_onnxruntime
    def test_qlinearconv(self):
        x = make_tensor_value_info('x', TensorProto.UINT8, [None, None, None, None])
        w = make_tensor_value_info('w', TensorProto.UINT8, [None, None, None, None])
        y = make_tensor_value_info('y', TensorProto.UINT8, [None, None, None, None])
        x_scale = make_tensor_value_info('x_scale', TensorProto.FLOAT, [None])
        w_scale = make_tensor_value_info('w_scale', TensorProto.FLOAT, [None])
        y_scale = make_tensor_value_info('y_scale', TensorProto.FLOAT, [None])
        x_zero_point = make_tensor_value_info('x_zero_point', TensorProto.UINT8, [None])
        w_zero_point = make_tensor_value_info('w_zero_point', TensorProto.UINT8, [None])
        y_zero_point = make_tensor_value_info('y_zero_point', TensorProto.UINT8, [None])
        node = make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], ['y'])
        graph = make_graph([node], 'g', [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point], [y])
        onnx_model = make_model_gen_version(graph, opset_imports=[make_opsetid('', 16)])
        sess1 = run_ort_inference(onnx_model)
        if sess1 is None:
            return
        sess2 = ReferenceEvaluator(onnx_model)
        sH, sW = (3, 3)
        for i in range(sH):
            for j in range(sW):
                x = np.zeros((1, 1, sH, sW), dtype=np.uint8)
                x[0, 0, i, j] = 1.0
                with self.subTest(w='1x1', i=i, j=j):
                    w = np.zeros((1, 1, 1, 1), dtype=np.uint8)
                    w[0, 0, :, :] = 1
                    feeds = {'x': x, 'x_scale': np.array([1], dtype=np.float32), 'x_zero_point': np.array([0], dtype=np.uint8), 'w': w, 'w_scale': np.array([1], dtype=np.float32), 'w_zero_point': np.array([0], dtype=np.uint8), 'y_scale': np.array([1], dtype=np.float32), 'y_zero_point': np.array([0], np.uint8)}
                    expected = sess1.run(None, feeds)[0]
                    got = sess2.run(None, feeds)[0]
                    try:
                        assert_allclose(expected, got)
                    except AssertionError as e:
                        raise e
                with self.subTest(w='3x3', i=i, j=j):
                    w = np.zeros((1, 1, 3, 3), dtype=np.uint8)
                    w[0, 0, :, :] = np.minimum(2 ** np.arange(9).reshape((3, -1)), 128)
                    feeds = {'x': x, 'x_scale': np.array([1], dtype=np.float32), 'x_zero_point': np.array([0], dtype=np.uint8), 'w': w, 'w_scale': np.array([1], dtype=np.float32), 'w_zero_point': np.array([0], dtype=np.uint8), 'y_scale': np.array([1], dtype=np.float32), 'y_zero_point': np.array([0], np.uint8)}
                    expected = sess1.run(None, feeds)[0]
                    got = sess2.run(None, feeds)[0]
                    assert_allclose(expected, got)
                with self.subTest(w='1x1', i=i, j=j):
                    w = np.zeros((1, 1, 1, 1), dtype=np.uint8)
                    w[0, 0, :, :] = 0
                    feeds = {'x': x, 'x_scale': np.array([0.00369204697], dtype=np.float32), 'x_zero_point': np.array([132], dtype=np.uint8), 'w': w, 'w_scale': np.array([100.00172794575], dtype=np.float32), 'w_zero_point': np.array([255], dtype=np.uint8), 'y_scale': np.array([0.00162681262], dtype=np.float32), 'y_zero_point': np.array([132], np.uint8)}
                    expected = sess1.run(None, feeds)[0]
                    got = sess2.run(None, feeds)[0]
                    assert_allclose(expected, got)
        x = np.array([[255, 174, 162, 25, 203, 168, 58], [15, 59, 237, 95, 129, 0, 64], [56, 242, 153, 221, 168, 12, 166], [232, 178, 186, 195, 237, 162, 237], [188, 39, 124, 77, 80, 102, 43], [127, 230, 21, 83, 41, 40, 134], [255, 154, 92, 141, 42, 148, 247]], dtype=np.uint8).reshape((1, 1, 7, 7))
        x_scale = np.array([0.00369204697], dtype=np.float32)
        x_zero_point = np.array([132], dtype=np.uint8)
        w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))
        w_scale = np.array([0.00172794575], dtype=np.float32)
        w_zero_point = np.array([255], dtype=np.uint8)
        y_scale = np.array([0.00162681262], dtype=np.float32)
        y_zero_point = np.array([123], dtype=np.uint8)
        feeds = {'x': x, 'x_scale': x_scale, 'x_zero_point': x_zero_point, 'w': w, 'w_scale': w_scale, 'w_zero_point': w_zero_point, 'y_scale': y_scale, 'y_zero_point': y_zero_point}
        expected = sess1.run(None, feeds)[0]
        got = sess2.run(None, feeds)[0]
        assert_allclose(expected, got)

    def common_test_im2col(self, kernel_shape, pads, strides, dilations):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
        Y1 = make_tensor_value_info('Y1', TensorProto.FLOAT, [None, None, None, None])
        Y2 = make_tensor_value_info('Y2', TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info('W', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('Conv', ['X', 'W'], ['Y1'], pads=pads, strides=strides, dilations=dilations)
        node_shape = make_node('Shape', ['W'], ['shape'])
        node_im = make_node('Im2Col', ['X', 'shape'], ['xim'], pads=pads, strides=strides, dilations=dilations, domain='experimental')
        node_flat = make_node('Flatten', ['W'], ['wflat'])
        node_gem = make_node('MatMul', ['wflat', 'xim'], ['Y2'])
        graph = make_graph([node, node_shape, node_im, node_flat, node_gem], 'g', [X, W], [Y1, Y2])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16), make_opsetid('experimental', 1)])
        graph_conv = make_graph([node], 'g', [X, W], [Y1])
        onnx_model_conv = make_model_gen_version(graph_conv, opset_imports=[make_opsetid('', 16)])
        sess = ReferenceEvaluator(onnx_model)
        try:
            sess_conv = run_ort_inference(onnx_model_conv)
            if sess_conv is None:
                return
        except ImportError:
            sess_conv = None
        sH, sW = (7, 7)
        nker = np.prod(kernel_shape)
        for i in range(sH):
            for j in range(sW):
                X = np.zeros((1, 1, sH, sW), dtype=np.float32)
                X[0, 0, i, j] = 1.0
                W = np.zeros((1, 1, *kernel_shape), dtype=np.float32)
                W[0, 0, :, :] = np.minimum(2 ** np.arange(nker).reshape((kernel_shape[0], -1)), 256)
                got = sess.run(None, {'X': X, 'W': W})
                if sess_conv is not None:
                    ort_res = sess_conv.run(None, {'X': X, 'W': W})[0]
                    assert_allclose(got[1].ravel(), ort_res.ravel())
                try:
                    assert_allclose(got[0].ravel(), got[1].ravel())
                except AssertionError as e:
                    raise AssertionError(f'Discrepancies: pads={pads}, dilations={dilations}, strides={strides}, kernel_shape={kernel_shape}\n{got[0]}\n!=\n{got[1]}') from e

    def test_im2col_1x1(self):
        self.common_test_im2col((1, 1), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1])

    def test_im2col_2x2(self):
        self.common_test_im2col((2, 2), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1])

    def test_im2col_3x3(self):
        self.common_test_im2col((3, 3), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1])

    def test_im2col_3x3_pads(self):
        self.common_test_im2col((3, 3), pads=[0, 1, 2, 3], strides=[1, 1], dilations=[1, 1])

    def test_im2col_3x3_strides(self):
        self.common_test_im2col((3, 3), pads=[0, 1, 1, 1], strides=[1, 2], dilations=[1, 1])

    def test_im2col_5x5(self):
        self.common_test_im2col((5, 5), pads=[1, 1, 1, 2], strides=[1, 1], dilations=[1, 1])

    @skip_if_no_torch
    def test_col2im(self):
        import torch
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None])
        IS = make_tensor_value_info('I', TensorProto.INT64, [None])
        BS = make_tensor_value_info('B', TensorProto.INT64, [None])
        node = make_node('Col2Im', ['X', 'I', 'B'], ['Y'], pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1])
        graph = make_graph([node], 'g', [X, IS, BS], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        sess = ReferenceEvaluator(onnx_model)
        X = np.array([[[1.0, 6.0, 11.0, 16.0, 21.0], [2.0, 7.0, 12.0, 17.0, 22.0], [3.0, 8.0, 13.0, 18.0, 23.0], [4.0, 9.0, 14.0, 19.0, 24.0], [5.0, 0.0, 15.0, 20.0, 25.0]]]).astype(np.float32)
        image_shape = np.array([5, 5]).astype(np.int64)
        block_shape = np.array([1, 5]).astype(np.int64)
        fold = torch.nn.Fold(output_size=tuple(image_shape), kernel_size=block_shape)
        got = sess.run(None, {'X': X, 'B': block_shape, 'I': image_shape})
        output = fold(torch.from_numpy(X)).numpy()
        assert_allclose(output, got[0])

    def common_test_col2im(self, size, image_shape, block_shape, pads, strides, dilations):
        import torch
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None])
        IS = make_tensor_value_info('I', TensorProto.INT64, [None])
        BS = make_tensor_value_info('B', TensorProto.INT64, [None])
        node = make_node('Col2Im', ['X', 'I', 'B'], ['Y'], pads=pads, strides=strides, dilations=dilations)
        graph = make_graph([node], 'g', [X, IS, BS], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        sess = ReferenceEvaluator(onnx_model)
        fold = torch.nn.Fold(output_size=tuple(image_shape), kernel_size=tuple(block_shape), dilation=tuple(dilations), padding=min(pads), stride=tuple(strides))
        nker = np.prod(block_shape)
        for i in range(nker):
            for j in range(size):
                X = np.zeros((1, nker, size), dtype=np.float32)
                X[0, i, j] = 1.0
                i_shape = np.array(image_shape, dtype=np.int64)
                b_shape = np.array(block_shape, dtype=np.int64)
                output = fold(torch.from_numpy(X)).numpy()
                got = sess.run(None, {'X': X, 'B': b_shape, 'I': i_shape})
                assert_allclose(output, got[0])

    @skip_if_no_torch
    def test_col2im_2x3(self):
        self.common_test_col2im(10, (6, 4), (2, 3), pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1])

    @skip_if_no_torch
    def test_col2im_2x3_pads(self):
        self.common_test_col2im(28, (6, 4), (2, 3), pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1])

    def test_col2im_2d(self):
        data = np.zeros([6, 28], dtype=np.float32)
        data[0][0] = 1.0
        image_shape, kernel_shape, dilations, pads, stride = (np.array([6, 4]), (2, 3), np.array([1, 1]), np.array([1, 1, 1, 1]), np.array([1, 1]))
        r1 = _col2im_naive_implementation_2d(data, image_shape, kernel_shape, dilations, pads, stride)
        r2 = col2im_naive_implementation(data, image_shape, kernel_shape, dilations, pads, stride)
        assert_allclose(r1, r2)

    def test_conv_im2col_group4(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [2, 4, 6, 6])
        W = make_tensor_value_info('W', TensorProto.FLOAT, [4, 1, 3, 3])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [4])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [2, 4, 6, 6])
        node = make_node('Conv', ['X', 'W', 'B'], ['Y'], group=4, dilations=[1, 1], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
        graph = make_graph([node], 'g', [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        feeds = {'X': np.arange(2 * 4 * 6 * 6).reshape((2, 4, 6, 6)).astype(np.float32), 'W': np.array([[[[-0.026239916682243347, 0.07565222680568695, -0.03209298849105835], [-0.08708783239126205, 0.0961190015077591, 0.13418219983577728], [0.1598859578371048, 0.03840477764606476, -0.13170936703681946]]], [[[-0.0689004510641098, 0.1408083587884903, -0.03717087209224701], [0.030967697501182556, 0.0263785719871521, -0.0899493545293808], [0.07828782498836517, -0.06266771256923676, 0.10750330984592438]]], [[[0.020227551460266113, -0.04353883117437363, -0.10938453674316406], [-0.14101561903953552, -0.03393106162548065, 0.12139306962490082], [0.02838282287120819, 0.13864465057849884, -0.06065710633993149]]], [[[-0.06511610746383667, -0.05987360328435898, -0.008047685027122498], [0.07340313494205475, 0.0326494425535202, 0.012516498565673828], [0.13260947167873383, -0.022225692868232727, -0.11167611926794052]]]], dtype=np.float32), 'B': np.array([-0.1457933485507965, -0.07481209933757782, -0.05890338122844696, -0.11964251846075058], dtype=np.float32)}
        feeds['B'][:] = 0
        X = feeds['X']
        W = feeds['W']
        B = feeds['B']
        Y = np.empty((2, 4, 6, 6), dtype=X.dtype)
        for b in range(X.shape[0]):
            for g in range(4):
                x = X[b:b + 1, g:g + 1]
                w = W[g]
                c2 = im2col(x, (3, 3), [1, 1], [1, 1, 1, 1], [1, 1])
                mul = np.matmul(c2, w.flatten())
                mul = mul + B[g]
                Y[b, g, :, :] = mul
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(Y, got1[0], atol=1e-05)

    def test_conv_strides(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 6, 6])
        W = make_tensor_value_info('W', TensorProto.FLOAT, [2, 3, 3, 3])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [2])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('Conv', ['X', 'W', 'B'], ['Y'], group=1, dilations=[1, 1], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2])
        graph = make_graph([node], 'g', [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        feeds = {'X': np.arange(1 * 3 * 6 * 6).reshape((1, 3, 6, 6)).astype(np.float32) + 1, 'W': np.zeros((2, 3, 3, 3), dtype=np.float32), 'B': np.zeros((2,), dtype=np.float32)}
        feeds['W'][0, 0, 0, 1] = 1
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([[[[0.0, 0.0, 0.0], [7.0, 9.0, 11.0], [19.0, 21.0, 23.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32)
        assert_allclose(expected, got1[0])

    def test_max_pool_2d_1(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('MaxPool', ['X'], ['Y'], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2])
        graph = make_graph([node], 'g', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        feeds = {'X': np.arange(49)[::-1].reshape((1, 1, 7, 7)).astype(np.float32)}
        expected = np.array([[[[48.0, 47.0, 45.0, 43.0], [41.0, 40.0, 38.0, 36.0], [27.0, 26.0, 24.0, 22.0], [13.0, 12.0, 10.0, 8.0]]]], dtype=np.float32)
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    def test_max_pool_2d_2(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('MaxPool', ['X'], ['Y'], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2])
        graph = make_graph([node], 'g', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        feeds = {'X': np.array([[[[683, 358, 726, 578, 650, 946, 200], [679, 260, 264, 5, 240, 255, 582], [322, 66, 687, 632, 852, 698, 428], [111, 452, 627, 332, 751, 842, 685], [472, 52, 956, 81, 807, 827, 360], [972, 574, 81, 799, 646, 499, 486], [892, 758, 75, 833, 972, 415, 736]]]], dtype=np.float32)}
        expected = np.array([[[[683.0, 726.0, 946.0, 946.0], [679.0, 687.0, 852.0, 842.0], [972.0, 956.0, 842.0, 842.0], [972.0, 833.0, 972.0, 736.0]]]], dtype=np.float32)
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    def test_scatter_elements(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Ind = make_tensor_value_info('I', TensorProto.INT64, [None, None])
        U = make_tensor_value_info('U', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        node = make_node('ScatterElements', ['X', 'I', 'U'], ['Y'], axis=1, reduction='min')
        graph = make_graph([node], 'g', [X, Ind, U], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        feeds = {'X': np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32), 'I': np.array([[1, 1]]), 'U': np.array([[1.1, 2.1]], dtype=np.float32)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([[1.0, 1.1, 3.0, 4.0, 5.0]], dtype=np.float32)
        assert_allclose(expected, got1[0])

    def test_scatternd(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Ind = make_tensor_value_info('I', TensorProto.INT64, [None, None])
        U = make_tensor_value_info('U', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        node = make_node('ScatterND', ['X', 'I', 'U'], ['Y'])
        graph = make_graph([node], 'g', [X, Ind, U], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        feeds = {'X': np.array([[1.0, 2.0]], dtype=np.float32), 'I': np.array([[0, 0]]), 'U': np.array([3.0], dtype=np.float32)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([[3.0, 2.0]], dtype=np.float32)
        assert_allclose(expected, got1[0])

    def test_col2im_impl(self):

        def get_im2col_indices(x_shape, field_height, field_width, padding=None, stride=1):
            N, C, H, W = x_shape
            del N
            assert (H + padding[0] + padding[2] - field_height) % stride == 0
            assert (W + padding[1] + padding[3] - field_height) % stride == 0
            out_height = (H + padding[0] + padding[2] - field_height) // stride + 1
            out_width = (W + padding[1] + padding[3] - field_width) // stride + 1
            i0 = np.repeat(np.arange(field_height), field_width)
            i0 = np.tile(i0, C)
            i1 = stride * np.repeat(np.arange(out_height), out_width)
            j0 = np.tile(np.arange(field_width), field_height * C)
            j1 = stride * np.tile(np.arange(out_width), out_height)
            i = i0.reshape(-1, 1) + i1.reshape(1, -1)
            j = j0.reshape(-1, 1) + j1.reshape(1, -1)
            k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
            return (k, i, j)

        def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=None, stride=1):
            N, C, H, W = x_shape
            H_padded, W_padded = (H + padding[0] + padding[2], W + padding[1] + padding[3])
            x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
            k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
            cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
            cols_reshaped = cols_reshaped.transpose(2, 0, 1)
            np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
            padding = padding.copy()
            if padding[2] == 0:
                padding[2] += x_padded.shape[2]
            elif padding[2] > 0:
                padding[2] *= -1
            if padding[3] == 0:
                padding[3] += x_padded.shape[3]
            elif padding[3] > 0:
                padding[3] *= -1
            res = x_padded[:, :, padding[0]:padding[2], padding[1]:padding[3]]
            return res
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None])
        IS = make_tensor_value_info('IS', TensorProto.INT64, [None])
        BS = make_tensor_value_info('BS', TensorProto.INT64, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('Col2Im', ['X', 'IS', 'BS'], ['Y'], pads=[0, 1, 0, 1])
        graph = make_graph([node], 'g', [X, IS, BS], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        feeds = {'X': np.arange(5 * 15).astype(np.float32).reshape((1, 5, 15)), 'IS': np.array([5, 5]), 'BS': np.array([1, 5])}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = col2im_indices(feeds['X'], (1, 1, 5, 5), field_height=1, field_width=5, padding=[0, 1, 0, 1])
        assert_allclose(expected, got1[0])

    def test_conv_transpose_2d(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info('W', TensorProto.FLOAT, [None, None, None, None])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('ConvTranspose', ['X', 'W', 'B'], ['Y'], dilations=[1, 1], kernel_shape=[3, 3], output_padding=[0, 0], pads=[1, 1, 1, 1], strides=[1, 1])
        graph = make_graph([node], 'g', [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        feeds = {'X': np.arange(1 * 3 * 5 * 4).reshape((1, 3, 5, 4)).astype(np.float32), 'W': np.arange(3 * 1 * 3 * 3).reshape((3, 1, 3, 3)).astype(np.float32), 'B': np.array([0, 0, 0, 0], dtype=np.float32)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([[[[4371, 6855, 7062, 4929], [7524, 11781, 12132, 8451], [8424, 13185, 13536, 9423], [9324, 14589, 14940, 10395], [7197, 11229, 11490, 7971]]]], dtype=np.float32)
        assert_allclose(expected, got1[0])
        feeds['X'] *= 0
        feeds['X'][0, 0, 0, 0] = 1
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([[[[4, 5, 0, 0], [7, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]], dtype=np.float32)
        assert_allclose(expected, got1[0])

    def test_conv_transpose_2d_upper(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
        W = make_tensor_value_info('W', TensorProto.FLOAT, [None, None, None, None])
        B = make_tensor_value_info('B', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('ConvTranspose', ['X', 'W', 'B'], ['Y'], auto_pad='SAME_UPPER', strides=[2, 2])
        graph = make_graph([node], 'g', [X, W, B], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
        feeds = {'X': np.arange(1 * 1 * 3 * 3).reshape((1, 1, 3, 3)).astype(np.float32), 'W': np.arange(1 * 2 * 3 * 3).reshape((1, 2, 3, 3)).astype(np.float32), 'B': np.array([0, 0, 0, 0], dtype=np.float32)}
        expected = np.array([[[[0, 0, 0, 1, 2, 2], [0, 0, 3, 4, 11, 8], [0, 3, 12, 11, 28, 19], [9, 12, 27, 16, 35, 20], [18, 27, 60, 35, 76, 43], [18, 24, 51, 28, 59, 32]], [[0, 0, 9, 10, 29, 20], [0, 0, 12, 13, 38, 26], [27, 30, 84, 56, 136, 82], [36, 39, 90, 52, 116, 65], [99, 108, 240, 134, 292, 160], [72, 78, 168, 91, 194, 104]]]], dtype=np.float32)
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    @unittest.skipIf(version_utils.numpy_older_than('1.21.5'), 'op_dft and op_stft requires numpy >= 1.21.5')
    def test_stft(self):
        signal = make_tensor_value_info('signal', TensorProto.FLOAT, [None, None, None])
        frame_step = make_tensor_value_info('frame_step', TensorProto.INT64, [None])
        frame_length = make_tensor_value_info('frame_length', TensorProto.INT64, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('STFT', ['signal', 'frame_step', '', 'frame_length'], ['Y'])
        graph = make_graph([node], 'g', [signal, frame_step, frame_length], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 17)])
        feeds = {'signal': np.arange(128).reshape((1, 128, 1)).astype(np.float32), 'frame_step': np.array(8, dtype=np.int64), 'frame_length': np.array(16, dtype=np.int64)}
        signal = feeds['signal']
        frame_length = int(feeds['frame_length'])
        frame_step = int(feeds['frame_step'])
        onesided_length = frame_length // 2 + 1
        nstfts = (feeds['signal'].shape[1] - frame_length) // frame_step + 1
        expected = np.empty([1, nstfts, onesided_length, 2], dtype=np.float32)
        for i in range(nstfts):
            start = i * frame_step
            stop = i * frame_step + frame_length
            complex_out = np.fft.fft(signal[0, start:stop, 0])
            c_out = complex_out[0:onesided_length]
            expected[0, i] = np.stack((c_out.real, c_out.imag), axis=1)
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    @unittest.skipIf(version_utils.numpy_older_than('1.21.5'), 'op_dft and op_stft requires numpy >= 1.21.5')
    def test_stft_with_window(self):
        signal = make_tensor_value_info('signal', TensorProto.FLOAT, [None, None, None])
        frame_step = make_tensor_value_info('frame_step', TensorProto.INT64, [None])
        window = make_tensor_value_info('window', TensorProto.FLOAT, [None])
        frame_length = make_tensor_value_info('frame_length', TensorProto.INT64, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        node = make_node('STFT', ['signal', 'frame_step', 'window', 'frame_length'], ['Y'])
        graph = make_graph([node], 'g', [signal, frame_step, window, frame_length], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 17)])
        feeds = {'signal': np.arange(128).reshape((1, 128, 1)).astype(np.float32), 'frame_step': np.array(8, dtype=np.int64), 'window': 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(0, 16, 1, dtype=np.float32) / 16), 'frame_length': np.array(16, dtype=np.int64)}
        signal = feeds['signal']
        frame_length = int(feeds['frame_length'])
        window = feeds['window']
        frame_step = int(feeds['frame_step'])
        onesided_length = frame_length // 2 + 1
        nstfts = 1 + (signal.shape[1] - window.shape[0]) // 8
        expected = np.empty([1, nstfts, onesided_length, 2], dtype=np.float32)
        for i in range(nstfts):
            start = i * frame_step
            stop = i * frame_step + frame_length
            complex_out = np.fft.fft(signal[0, start:stop, 0] * window)[0:onesided_length]
            c_out = complex_out[0:onesided_length]
            expected[0, i] = np.stack((c_out.real, c_out.imag), axis=1)
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        assert_allclose(expected, got1[0])

    def get_roi_align_model(self, mode):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
        rois = make_tensor_value_info('rois', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        IS = make_tensor_value_info('I', TensorProto.INT64, [None])
        node = make_node('RoiAlign', ['X', 'rois', 'I'], ['Y'], output_height=5, output_width=5, sampling_ratio=2, spatial_scale=1.0, coordinate_transformation_mode='output_half_pixel', mode=mode)
        graph = make_graph([node], 'g', [X, rois, IS], [Y])
        return make_model_gen_version(graph, opset_imports=[make_opsetid('', 17)])

    def common_test_roi_align(self, mode):
        onnx_model = self.get_roi_align_model(mode)
        X, batch_indices, rois = get_roi_align_input_values()
        feeds = {'X': X, 'rois': rois, 'I': batch_indices}
        sess = run_ort_inference(onnx_model)
        if sess is None:
            return
        expected = sess.run(None, feeds)
        ref = ReferenceEvaluator(onnx_model)
        got = ref.run(None, feeds)
        assert_allclose(expected[0], got[0], atol=1e-05)

    @skip_if_no_onnxruntime
    def test_roi_align(self):
        with self.subTest(mode='avg'):
            self.common_test_roi_align('avg')
        with self.subTest(mode='max'):
            self.common_test_roi_align('max')

    def common_test_roi_align_torch(self, mode):
        import torch
        from torchvision.ops import RoIAlign
        onnx_model = self.get_roi_align_model(mode)
        sess = ReferenceEvaluator(onnx_model)
        X, batch_indices, rois = get_roi_align_input_values()
        got = sess.run(None, {'X': X, 'rois': rois, 'I': batch_indices})
        a = RoIAlign((5, 5), spatial_scale=1.0, sampling_ratio=2)
        expected = a(torch.from_numpy(X), [torch.from_numpy(rois)])
        assert_allclose(expected, got[0], atol=1e-05)

    @skip_if_no_torch
    @skip_if_no_torchvision
    def test_roi_align_torch(self):
        with self.subTest(mode='avg'):
            self.common_test_roi_align_torch('avg')

    def test_split(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y1 = make_tensor_value_info('Y1', TensorProto.FLOAT, [None])
        Y2 = make_tensor_value_info('Y2', TensorProto.FLOAT, [None])
        Y3 = make_tensor_value_info('Y3', TensorProto.FLOAT, [None])
        Y4 = make_tensor_value_info('Y4', TensorProto.FLOAT, [None])
        node = make_node('Split', ['X'], ['Y1', 'Y2', 'Y3', 'Y4'], num_outputs=4)
        graph = make_graph([node], 'g', [X], [Y1, Y2, Y3, Y4])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        feeds = {'X': np.arange(10).astype(np.float32)}
        expected = [np.array([0, 1, 2], dtype=np.float32), np.array([3, 4, 5], dtype=np.float32), np.array([6, 7, 8], dtype=np.float32), np.array([9], dtype=np.float32)]
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        for i in range(4):
            assert_allclose(expected[i], got1[i])

    def test_split_2(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y1 = make_tensor_value_info('Y1', TensorProto.FLOAT, [None])
        Y2 = make_tensor_value_info('Y2', TensorProto.FLOAT, [None])
        Y3 = make_tensor_value_info('Y3', TensorProto.FLOAT, [None])
        Y4 = make_tensor_value_info('Y4', TensorProto.FLOAT, [None])
        node = make_node('Split', ['X', 'split'], ['Y1', 'Y2', 'Y3', 'Y4'])
        graph = make_graph([node], 'g', [X], [Y1, Y2, Y3, Y4])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        feeds = {'X': np.arange(10).astype(np.float32), 'split': np.array([3, 3, 2, 2], dtype=np.int64)}
        expected = [np.array([0, 1, 2], dtype=np.float32), np.array([3, 4, 5], dtype=np.float32), np.array([6, 7], dtype=np.float32), np.array([8, 9], dtype=np.float32)]
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        for i in range(4):
            assert_allclose(expected[i], got1[i])

    def test_split_num_outputs_4(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y1 = make_tensor_value_info('Y1', TensorProto.FLOAT, [None])
        Y2 = make_tensor_value_info('Y2', TensorProto.FLOAT, [None])
        Y3 = make_tensor_value_info('Y3', TensorProto.FLOAT, [None])
        Y4 = make_tensor_value_info('Y4', TensorProto.FLOAT, [None])
        node = make_node('Split', ['X'], ['Y1', 'Y2', 'Y3', 'Y4'], num_outputs=4)
        graph = make_graph([node], 'g', [X], [Y1, Y2, Y3, Y4])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        feeds = {'X': np.arange(10).astype(np.float32)}
        expected = [np.array([0, 1, 2], dtype=np.float32), np.array([3, 4, 5], dtype=np.float32), np.array([6, 7, 8], dtype=np.float32), np.array([9], dtype=np.float32)]
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        for i in range(4):
            assert_allclose(expected[i], got1[i])
        feeds = {'X': np.arange(9).astype(np.float32)}
        expected = [np.array([0, 1, 2], dtype=np.float32), np.array([3, 4, 5], dtype=np.float32), np.array([6, 7, 8], dtype=np.float32), np.array([], dtype=np.float32)]
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        for i in range(4):
            assert_allclose(expected[i], got1[i])

    def test_argmin(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.INT64, [None])
        node = make_node('ArgMin', ['X'], ['Y'], axis=1)
        graph = make_graph([node], 'g', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        feeds = {'X': np.arange(12).reshape((3, 4)).astype(np.float32)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([0, 0, 0], dtype=np.int64).reshape((-1, 1))
        self.assertEqual(expected.tolist(), got1[0].tolist())

    def test_argmax(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.INT64, [None])
        node = make_node('ArgMax', ['X'], ['Y'], axis=1)
        graph = make_graph([node], 'g', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        feeds = {'X': np.arange(12).reshape((3, 4)).astype(np.float32)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([3, 3, 3], dtype=np.int64).reshape((-1, 1))
        self.assertEqual(expected.tolist(), got1[0].tolist())

    def test_slice_squeeze(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        starts = make_tensor_value_info('starts', TensorProto.INT64, [None])
        ends = make_tensor_value_info('ends', TensorProto.INT64, [None])
        axes = make_tensor_value_info('axes', TensorProto.INT64, [None])
        Y = make_tensor_value_info('Y', TensorProto.INT64, [None])
        nodes = [make_node('Slice', ['X', 'starts', 'ends', 'axes'], ['T']), make_node('Squeeze', ['T', 'axes'], ['Y'])]
        graph = make_graph(nodes, 'g', [X, starts, ends, axes], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        feeds = {'X': np.array([[0]], dtype=np.int64), 'starts': np.array([0], dtype=np.int64), 'ends': np.array([1], dtype=np.int64), 'axes': np.array([0], dtype=np.int64)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([0], dtype=np.int64)
        self.assertEqual(expected.tolist(), got1[0].tolist())

    def test_slice_squeeze_6(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.INT64, [None])
        nodes = [make_node('Slice', ['X'], ['T'], axes=[0], starts=[0], ends=[1]), make_node('Squeeze', ['T'], ['Y'], axes=[0])]
        graph = make_graph(nodes, 'g', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 6)])
        feeds = {'X': np.array([[0]], dtype=np.int64)}
        ref1 = ReferenceEvaluator(onnx_model)
        got1 = ref1.run(None, feeds)
        expected = np.array([0], dtype=np.int64)
        self.assertEqual(expected.tolist(), got1[0].tolist())

    def test_onnxrt_reduce_mean(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
        node1 = make_node('ReduceMean', ['X'], ['Y'])
        graph = make_graph([node1], 'g', [X], [Y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 17)])
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        cls = sess.rt_nodes_[0]
        self.assertEqual(cls.__class__.__name__, 'ReduceMean_1')
        got = sess.run(None, {'X': np.ones((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (1, 1))
        self.assertEqual(got[0, 0], 1)
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
        check_model(onnx_model)
        sess = ReferenceEvaluator(onnx_model)
        cls = sess.rt_nodes_[0]
        self.assertEqual(cls.__class__.__name__, 'ReduceMean_18')
        got = sess.run(None, {'X': np.ones((2, 4), dtype=np.float32)})[0]
        self.assertEqual(got.shape, (1, 1))
        self.assertEqual(got[0, 0], 1)

    @staticmethod
    def _cdist_model(opset, reduce_op='ReduceSumSquare'):
        initializers = []
        inputs = [make_tensor_value_info('next_in', TensorProto.FLOAT, [None, 4]), make_tensor_value_info('next', TensorProto.FLOAT, [None])]
        outputs = [make_tensor_value_info('next_out', TensorProto.FLOAT, [None, None]), make_tensor_value_info('scan_out', TensorProto.FLOAT, [None])]
        if opset >= 18:
            initializers.append(from_array(np.array([1], dtype=np.int64), name='axis_red'))
            node_reduce = make_node(reduce_op, ['cdistdf_17_C0', 'axis_red'], ['cdistdf_17_reduced0'], name='cdistdf_17_ReduceSumSquare', keepdims=0)
        else:
            node_reduce = make_node(reduce_op, ['cdistdf_17_C0'], ['cdistdf_17_reduced0'], name='cdistdf_17_ReduceSumSquare', axes=[1], keepdims=0)
        nodes = [make_node('Identity', ['next_in'], ['next_out'], name='cdistd_17_Identity'), make_node('Sub', ['next_in', 'next'], ['cdistdf_17_C0'], name='cdistdf_17_Sub'), node_reduce, make_node('Identity', ['cdistdf_17_reduced0'], ['scan_out'], name='cdistdf_17_Identity')]
        graph = make_graph(nodes, 'OnnxIdentity', inputs, outputs, initializers)
        initializers = []
        list_value = [1.1394007205963135, -0.6848101019859314, -1.234825849533081, 0.4023416340351105, 0.17742614448070526, 0.46278226375579834, -0.4017809331417084, -1.630198359489441, -0.5096521973609924, 0.7774903774261475, -0.4380742907524109, -1.2527953386306763, -1.0485529899597168, 1.950775384902954, -1.420017957687378, -1.7062702178955078, 1.8675580024719238, -0.15135720372200012, -0.9772778749465942, 0.9500884413719177, -2.5529897212982178, -0.7421650290489197, 0.653618574142456, 0.8644362092018127, 1.5327792167663574, 0.37816253304481506, 1.4693588018417358, 0.154947429895401, -0.6724604368209839, -1.7262825965881348, -0.35955315828323364, -0.8131462931632996, -0.8707971572875977, 0.056165341287851334, -0.5788496732711792, -0.3115525245666504, 1.2302906513214111, -0.302302747964859, 1.202379822731018, -0.38732680678367615, 2.269754648208618, -0.18718385696411133, -1.4543657302856445, 0.04575851559638977, -0.9072983860969543, 0.12898291647434235, 0.05194539576768875, 0.7290905714035034, 1.4940791130065918, -0.8540957570075989, -0.2051582634449005, 0.3130677044391632, 1.764052391052246, 2.2408931255340576, 0.40015721321105957, 0.978738009929657, 0.06651721894741058, -0.3627411723136902, 0.30247190594673157, -0.6343221068382263, -0.5108051300048828, 0.4283318817615509, -1.18063223361969, -0.02818222902715206, -1.6138978004455566, 0.38690251111984253, -0.21274028718471527, -0.8954665660858154, 0.7610377073287964, 0.3336743414402008, 0.12167501449584961, 0.44386324286460876, -0.10321885347366333, 1.4542734622955322, 0.4105985164642334, 0.14404356479644775, -0.8877857327461243, 0.15634897351264954, -1.980796456336975, -0.34791216254234314]
        initializers.append(from_array(np.array(list_value, dtype=np.float32).reshape((20, 4)), name='Sc_Scancst'))
        initializers.append(from_array(np.array([2], dtype=np.int64), name='To_TopKcst'))
        inputs = [make_tensor_value_info('input', TensorProto.FLOAT, [None, 4])]
        outputs = [make_tensor_value_info('values', TensorProto.FLOAT, [None, 2]), make_tensor_value_info('indices', TensorProto.INT64, [None, 2])]
        nodes = [make_node('Scan', ['input', 'Sc_Scancst'], ['UU032UU', 'UU033UU'], name='Sc_Scan', body=graph, num_scan_inputs=1), make_node('Transpose', ['UU033UU'], ['Tr_transposed0'], name='Tr_Transpose', perm=[1, 0]), make_node('Sqrt', ['Tr_transposed0'], ['Sq_Y0'], name='Sq_Sqrt'), make_node('TopK', ['Sq_Y0', 'To_TopKcst'], ['values', 'indices'], name='To_TopK', largest=0, sorted=1)]
        graph = make_graph(nodes, 'dummy', inputs, outputs, initializers)
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', opset)])
        return onnx_model

    @parameterized.parameterized.expand(itertools.product([('ReduceMin', [np.array([[np.nan, np.nan], [14.422706, 18.80527]], dtype=np.float32), np.array([[2, 15], [10, 4]], dtype=np.int64)]), ('ReduceL1', [np.array([[2.2367053, 2.3516612], [4.076292, 4.2970634]], dtype=np.float32), np.array([[18, 6], [13, 6]], dtype=np.int64)]), ('ReduceL2', [np.array([[1.80155, 1.8169948], [2.9928076, 3.1205883]], dtype=np.float32), np.array([[11, 18], [13, 6]], dtype=np.int64)]), ('ReduceLogSum', [np.array([[0.9497848, 1.1872643], [1.6764175, 1.70759]], dtype=np.float32), np.array([[6, 18], [13, 6]], dtype=np.int64)]), ('ReduceLogSumExp', [np.array([[1.6005973, 1.7445935], [2.5616229, 2.6539795]], dtype=np.float32), np.array([[13, 6], [13, 6]], dtype=np.int64)]), ('ReduceMax', [np.array([[1.4217108, 1.5069536], [2.453826, 2.5041783]], dtype=np.float32), np.array([[13, 11], [13, 11]], dtype=np.int64)]), ('ReduceMean', [np.array([[0.39247903, 0.78497636], [2.038146, 2.1485317]], dtype=np.float32), np.array([[13, 6], [13, 6]], dtype=np.int64)]), ('ReduceSumSquare', [np.array([[3.2455828, 3.3014696], [8.956896, 9.7380705]], dtype=np.float32), np.array([[11, 18], [13, 6]], dtype=np.int64)]), ('ReduceProd', [np.array([[np.nan, np.nan], [14.422706, 18.80527]], dtype=np.float32), np.array([[2, 15], [13, 6]], dtype=np.int64)])], [17, 18]))
    def test_op_reduce(self, reduce_op_expected, opset: int):
        reduce_op, expected = reduce_op_expected
        X = np.arange(8).reshape((-1, 4)).astype(np.float32)
        results = {}
        model = self._cdist_model(opset, reduce_op)
        sess = ReferenceEvaluator(model)
        got = sess.run(None, {'input': X})
        results['ref', opset] = got
        cl = [n for n in sess.rt_nodes_[0].body.rt_nodes_ if n.__class__.__name__.startswith(reduce_op)]
        schema = cl[0]._schema
        new_cl = type(reduce_op, (cl[0].__class__,), {'op_schema': schema})
        sess = ReferenceEvaluator(model, new_ops=[new_cl])
        got = sess.run(None, {'input': X})
        results['ref_cl', opset] = got
        baseline = 'constant'
        for k, v in results.items():
            for a, b in zip(reversed(expected), reversed(v)):
                if a.shape != b.shape:
                    raise AssertionError(f'Shape mismatch for {reduce_op!r}, {baseline}:{a.shape} != {k}:{b.shape}.')
                diff = np.abs(a - b).max()
                if diff > 1e-06:
                    raise AssertionError(f'Discrepancies (max={diff}) for {reduce_op!r}, {baseline} != {k}\n{a}\n!=\n{b}')

    @parameterized.parameterized.expand([(13,), (17,), (18,)])
    def test_mvn(self, opset: int, ref_opset: int=13):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
        nodes = [make_node('MeanVarianceNormalization', ['X'], ['Y'])]
        graph = make_graph(nodes, 'g', [X], [Y])
        x = np.random.rand(3, 3, 3, 1).astype(np.float32)
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', opset)])
        ref = ReferenceEvaluator(onnx_model)
        got = ref.run(None, {'X': x})[0]
        ref_onnx_model = make_model(graph, opset_imports=[make_opsetid('', ref_opset)])
        ref_expected = ReferenceEvaluator(ref_onnx_model)
        expected = ref_expected.run(None, {'X': x})[0]
        self.assertEqual(expected.shape, got.shape)
        assert_allclose(expected, got)

    def test_concat_in_a_function(self):

        def create_model():
            nodes = []
            inputs = []
            outputs = []
            functions = []
            opsets = {'': onnx_opset_version(), 'custom_domain': 1}
            nodes_fct = []
            node = make_node('Concat', ['x:0', 'x:1'], ['r__0'], axis=0, domain='')
            nodes_fct.append(node)
            opset_imports_fct = [make_opsetid(domain, 1 if version is None else version) for domain, version in opsets.items()]
            fct = make_function('custom_domain', 'concat_2', ['x:0', 'x:1'], ['r__0'], nodes_fct, opset_imports_fct)
            functions.append(fct)
            inputs.append(make_tensor_value_info('I__0', TensorProto.DOUBLE, []))
            inputs.append(make_tensor_value_info('I__1', TensorProto.DOUBLE, []))
            inputs.append(make_tensor_value_info('I__2', TensorProto.DOUBLE, []))
            outputs.append(make_tensor_value_info('r__4', TensorProto.DOUBLE, []))
            node = make_node('concat_2', ['I__0', 'I__1'], ['r__3'], axis=0, domain='custom_domain')
            nodes.append(node)
            node = make_node('concat_2', ['I__2', 'r__3'], ['r__4'], axis=0, domain='custom_domain')
            nodes.append(node)
            opset_imports = [make_opsetid(domain, 1 if version is None else version) for domain, version in opsets.items()]
            graph = make_graph(nodes, 'numpyx', inputs, outputs)
            onnx_model = make_model(graph, opset_imports=opset_imports, functions=functions)
            return onnx_model
        onnx_model = create_model()
        x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
        x2 = np.array([[1, 2]], dtype=np.float64)
        x3 = np.array([[-1, -2]], dtype=np.float64)
        z = np.vstack([x1, x2, x3])
        ref = ReferenceEvaluator(onnx_model)
        feeds = {'I__2': x1, 'I__0': x2, 'I__1': x3}
        got = ref.run(None, feeds)
        assert_allclose(z, got[0])

    def test_cast_float_to_string(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.STRING, [None])
        model = make_model(make_graph([make_node('Cast', ['X'], ['Y'], to=TensorProto.STRING)], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        data = np.array([1.152512, -0.152612, 0.0, np.nan])
        got = ref.run(None, {'X': data})[0]
        self.assertTrue((got == np.array([1.152512, -0.152612, 0.0, np.nan]).astype(np.str_)).all())

    def test_cast_float_to_string_and_back(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        model = make_model(make_graph([make_node('Cast', ['X'], ['Z'], to=TensorProto.STRING), make_node('Cast', ['Z'], ['Y'], to=TensorProto.FLOAT)], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        data = np.array([1.152512, -0.152612, 0.0, np.nan])
        got = ref.run(None, {'X': data})[0]
        assert_allclose(got, np.array([1.152512, -0.152612, 0.0, np.nan]))

    def test_split_to_sequence(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, None)
        Y = make_tensor_value_info('Y', TensorProto.INT64, None)
        Z = make_tensor_value_info('Z', TensorProto.UNDEFINED, None)
        nodes = [make_node('SplitToSequence', ['X', 'Y'], ['Z'], axis=2)]
        model = make_model(make_graph(nodes, 'g', [X, Y], [Z]))
        ref = ReferenceEvaluator(model)
        data = np.arange(18).reshape((1, 3, 6)).astype(np.float32)
        indices = np.array(2, dtype=np.int64)
        got = ref.run(None, {'X': data, 'Y': indices})
        expected = [[np.array([[[0.0, 1.0], [6.0, 7.0], [12.0, 13.0]]], dtype=np.float32), np.array([[[2.0, 3.0], [8.0, 9.0], [14.0, 15.0]]], dtype=np.float32), np.array([[[4.0, 5.0], [10.0, 11.0], [16.0, 17.0]]], dtype=np.float32)]]
        self.assertEqual(len(expected[0]), len(got[0]))
        for a, b in zip(expected[0], got[0]):
            assert_allclose(a, b)

    def test_split_to_sequence_1d(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, None)
        Y = make_tensor_value_info('Y', TensorProto.INT64, None)
        Z = make_tensor_value_info('Z', TensorProto.UNDEFINED, None)
        nodes = [make_node('SplitToSequence', ['X', 'Y'], ['Z'], axis=2)]
        model = make_model(make_graph(nodes, 'g', [X, Y], [Z]))
        ref = ReferenceEvaluator(model)
        data = np.arange(18).reshape((1, 3, 6)).astype(np.float32)
        indices = np.array([2, 2, 2], dtype=np.int64)
        got = ref.run(None, {'X': data, 'Y': indices})
        expected = [[np.array([[[0.0, 1.0], [6.0, 7.0], [12.0, 13.0]]], dtype=np.float32), np.array([[[2.0, 3.0], [8.0, 9.0], [14.0, 15.0]]], dtype=np.float32), np.array([[[4.0, 5.0], [10.0, 11.0], [16.0, 17.0]]], dtype=np.float32)]]
        self.assertEqual(len(expected[0]), len(got[0]))
        for a, b in zip(expected[0], got[0]):
            assert_allclose(a, b)

    def test_split_to_sequence_nokeepdims_noinput(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, None)
        Z = make_tensor_value_info('Z', TensorProto.UNDEFINED, None)
        nodes = [make_node('SplitToSequence', ['X'], ['Z'], axis=2, keepdims=0)]
        model = make_model(make_graph(nodes, 'g', [X], [Z]))
        ref = ReferenceEvaluator(model)
        data = np.arange(18).reshape((1, 3, 6)).astype(np.float32)
        got = ref.run(None, {'X': data})
        expected = [[data[:, :, i] for i in range(data.shape[2])]]
        self.assertEqual(len(expected[0]), len(got[0]))
        for a, b in zip(expected[0], got[0]):
            assert_allclose(a, b)

    def test_cast_float8(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        F1 = make_tensor_value_info('F1', TensorProto.FLOAT, [None])
        F2 = make_tensor_value_info('F2', TensorProto.FLOAT, [None])
        F3 = make_tensor_value_info('F3', TensorProto.FLOAT, [None])
        F4 = make_tensor_value_info('F4', TensorProto.FLOAT, [None])
        model = make_model(make_graph([make_node('Cast', ['X'], ['f81'], to=TensorProto.FLOAT8E4M3FN), make_node('Cast', ['X'], ['f82'], to=TensorProto.FLOAT8E5M2), make_node('Constant', [], ['C1'], value=make_tensor('C1', TensorProto.FLOAT8E4M3FN, [5], [0, 1, 2, 0.05, 200])), make_node('Constant', [], ['C2'], value=make_tensor('C2', TensorProto.FLOAT8E5M2, [5], [0, 1, 2, 0.05, 200])), make_node('Cast', ['f81'], ['F1'], to=TensorProto.FLOAT), make_node('Cast', ['f82'], ['F2'], to=TensorProto.FLOAT), make_node('Cast', ['C1'], ['F3'], to=TensorProto.FLOAT), make_node('Cast', ['C2'], ['F4'], to=TensorProto.FLOAT)], 'g', [X], [F1, F2, F3, F4]))
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 0.05, 200], dtype=np.float32)
        expected1 = np.array([float8e4m3_to_float32(float32_to_float8e4m3(x)) for x in data])
        expected2 = np.array([float8e5m2_to_float32(float32_to_float8e5m2(x)) for x in data])
        got = ref.run(None, {'X': data})
        assert_allclose(got[0], expected1)
        assert_allclose(got[1], expected2)
        assert_allclose(got[2], expected1)
        assert_allclose(got[3], expected2)

    def test_cast_like_float8(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        model = make_model(make_graph([make_node('Cast', ['X'], ['f8'], to=TensorProto.FLOAT8E4M3FNUZ), make_node('CastLike', ['X', 'f8'], ['f32'], saturate=0), make_node('Cast', ['f32'], ['Y'], to=TensorProto.FLOAT)], 'g', [X], [Y]))
        data = np.array([0, 10000000.0], dtype=np.float32)
        expected = np.array([float8e4m3_to_float32(float32_to_float8e4m3(x, uz=True, saturate=False), uz=True) for x in data])
        ref = ReferenceEvaluator(model)
        got = ref.run(None, {'X': data})
        assert_allclose(got[0], expected)

        class CastLike(OpRunExpand):
            op_domain = ''
        ref = ReferenceEvaluator(model, new_ops=[CastLike])
        got = ref.run(None, {'X': data})
        assert_allclose(got[0], expected)

    def test_cast_float8_output(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        F1 = make_tensor_value_info('F1', TensorProto.FLOAT8E4M3FN, [None])
        F2 = make_tensor_value_info('F2', TensorProto.FLOAT8E5M2, [None])
        model = make_model(make_graph([make_node('Cast', ['X'], ['F1'], to=TensorProto.FLOAT8E4M3FN), make_node('Cast', ['X'], ['F2'], to=TensorProto.FLOAT8E5M2)], 'g', [X], [F1, F2]))
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 0.05, 200], dtype=np.float32)
        expected1 = np.array([float32_to_float8e4m3(x) for x in data])
        expected2 = np.array([float32_to_float8e5m2(x) for x in data])
        got = ref.run(None, {'X': data})
        self.assertEqual(expected1.tolist(), got[0].tolist())
        self.assertEqual(expected2.tolist(), got[1].tolist())

    def test_float8_4_types(self):
        x = np.array([0.4068359375, 352, 416, 336, 304, 272, -248, -100, 0.0001, 0.01, 416, 432, 100000.0, np.inf, -np.inf, np.nan], dtype=np.float32)
        expected = {TensorProto.FLOAT8E4M3FN: np.array([0.40625, 352.0, 416.0, 320.0, 320.0, 256.0, -256.0, -96.0, 0.0, 0.009765625, 416.0, 448.0, 448.0, 448.0, -448.0, np.nan], dtype=np.float32), TensorProto.FLOAT8E4M3FNUZ: np.array([0.40625, 240.0, 240.0, 240.0, 240.0, 240.0, -240.0, -96.0, 0.0, 0.009765625, 240.0, 240.0, 240.0, 240.0, -240.0, np.nan], dtype=np.float32), TensorProto.FLOAT8E5M2: np.array([0.4375, 384.0, 384.0, 320.0, 320.0, 256.0, -256.0, -96.0, 0.0001068115234375, 0.009765625, 384.0, 448.0, 57344.0, 57344.0, -57344.0, np.nan], dtype=np.float32), TensorProto.FLOAT8E5M2FNUZ: np.array([0.4375, 384.0, 384.0, 320.0, 320.0, 256.0, -256.0, -96.0, 0.00010681152, 0.009765625, 384.0, 448.0, 57344.0, 57344.0, -57344.0, np.nan], dtype=np.float32)}

        def model_cast_cast(to):
            X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
            Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
            node1 = make_node('Cast', ['X'], ['T'], to=to)
            node2 = make_node('Cast', ['T'], ['Y'], to=TensorProto.FLOAT)
            graph = make_graph([node1, node2], 'lr', [X], [Y])
            onnx_model = make_model(graph)
            check_model(onnx_model)
            return onnx_model
        for to, expect in expected.items():
            with self.subTest(to=to):
                onnx_model = model_cast_cast(to)
                ref = ReferenceEvaluator(onnx_model)
                y = ref.run(None, {'X': x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    def test_cast_bfloat16_output(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.BFLOAT16, [None])
        model = make_model(make_graph([make_node('Cast', ['X'], ['Y'], to=TensorProto.BFLOAT16)], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 100000.0, 200], dtype=np.float32)
        expected1 = np.array([float32_to_bfloat16(x) for x in data])
        got = ref.run(None, {'X': data})
        self.assertEqual(expected1.tolist(), got[0].tolist())

    def test_quantize_linear_e4m3(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        model = make_model(make_graph([make_node('Constant', [], ['scale'], value=make_tensor('scale', TensorProto.FLOAT, [1], [2.0])), make_node('Constant', [], ['zero'], value=make_tensor('zero', TensorProto.FLOAT8E4M3FN, [1], [0.0])), make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['T']), make_node('DequantizeLinear', ['T', 'scale'], ['Y'], axis=0)], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 100000.0, 200], dtype=np.float32)
        expected = np.array([0, 1, 2, 896, 192], dtype=np.float32)
        got = ref.run(None, {'X': data})
        assert_allclose(expected, got[0])

    def test_quantize_linear_e4m3_initializer(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        model = make_model(make_graph([make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['T']), make_node('DequantizeLinear', ['T', 'scale'], ['Y'], axis=0)], 'g', [X], [Y], [make_tensor('scale', TensorProto.FLOAT, [1], [2.0]), make_tensor('zero', TensorProto.FLOAT8E4M3FN, [1], [0.0])]))
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 100000.0, 200], dtype=np.float32)
        expected = np.array([0, 1, 2, 896, 192], dtype=np.float32)
        got = ref.run(None, {'X': data})
        assert_allclose(expected, got[0])

    def test_quantize_linear_e5m2(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        model = make_model(make_graph([make_node('Constant', [], ['scale'], value=make_tensor('scale', TensorProto.FLOAT, [1], [2.0])), make_node('Constant', [], ['zero'], value=make_tensor('zero', TensorProto.FLOAT8E5M2, [1], [0.0])), make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['T']), make_node('DequantizeLinear', ['T', 'scale'], ['Y'], axis=0)], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2, 100000.0, 200], dtype=np.float32)
        expected = np.array([0, 1, 2, 98304, 192], dtype=np.float32)
        got = ref.run(None, {'X': data})
        assert_allclose(expected, got[0])

    def test_quantize_linear_uint16(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.UINT16, [None])
        model = make_model(make_graph([make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['Y'])], 'g', [X], [Y], [make_tensor('scale', TensorProto.FLOAT, [1], [2.0]), make_tensor('zero', TensorProto.UINT16, [1], [32767])]))
        ref = ReferenceEvaluator(model)
        data = np.array([0.0, -128.0, 3.0, -3.0, 2.9, -2.9, 3.1, -3.1, 65536.0, -65534.0, 70000.0, -70000.0], dtype=np.float32)
        expected = np.array([32767, 32703, 32769, 32765, 32768, 32766, 32769, 32765, 65535, 0, 65535, 0], dtype=np.uint16)
        got = ref.run(None, {'X': data})
        assert_allclose(expected, got[0])

    def test_quantize_linear_int16(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.INT16, [None])
        model = make_model(make_graph([make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['Y'])], 'g', [X], [Y], [make_tensor('scale', TensorProto.FLOAT, [1], [2.0]), make_tensor('zero', TensorProto.INT16, [1], [256])]))
        ref = ReferenceEvaluator(model)
        data = np.array([0.0, -514.0, 3.0, -3.0, 2.9, -2.9, 3.1, -3.1, 65022.0, -66046.0, 65023.0, -66047.0, 65024.0, -66048.0, 70000.0, -70000.0], dtype=np.float32)
        expected = np.array([256, -1, 258, 254, 257, 255, 258, 254, 32767, -32767, 32767, -32768, 32767, -32768, 32767, -32768], dtype=np.int16)
        got = ref.run(None, {'X': data})
        assert_allclose(expected, got[0])

    def test_dequantize_linear_uint16(self):
        X = make_tensor_value_info('X', TensorProto.UINT16, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        model = make_model(make_graph([make_node('DequantizeLinear', ['X', 'scale', 'zero'], ['Y'], axis=0)], 'g', [X], [Y], [make_tensor('scale', TensorProto.FLOAT, [1], [2.0]), make_tensor('zero', TensorProto.UINT16, [1], [32767])]))
        ref = ReferenceEvaluator(model)
        data = np.array([30000, 31000, 32768, 33000], dtype=np.uint16)
        expected = np.array([-5534.0, -3534.0, 2.0, 466.0], dtype=np.float32)
        got = ref.run(None, {'X': data})
        assert_allclose(expected, got[0])

    def test_dequantize_linear_int16(self):
        X = make_tensor_value_info('X', TensorProto.INT16, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        model = make_model(make_graph([make_node('DequantizeLinear', ['X', 'scale', 'zero'], ['Y'], axis=0)], 'g', [X], [Y], [make_tensor('scale', TensorProto.FLOAT, [1], [2.0]), make_tensor('zero', TensorProto.INT16, [1], [-1024])]))
        ref = ReferenceEvaluator(model)
        data = np.array([-300, -30, -1025, 1270], dtype=np.int16)
        expected = np.array([1448.0, 1988.0, -2.0, 4588.0], dtype=np.float32)
        got = ref.run(None, {'X': data})
        assert_allclose(expected, got[0])

    @parameterized.parameterized.expand([(4 * np.arange(12).reshape(3, 4), np.arange(1, 7).reshape(3, 2), np.zeros((3, 2)), 1, 2, [[0, 4, 4, 6], [5, 7, 6, 7], [6, 7, 7, 7]]), (4 * np.arange(12).reshape(3, 4), np.arange(1, 7).reshape(3, 2), np.ones((3, 2)), 1, 2, [[1, 5, 5, 7], [6, 8, 7, 8], [7, 8, 8, 8]]), (np.arange(24).reshape(3, 8), [[0.25, 0.5, 1], [0.25, 0.5, 1], [0.25, 0.5, 1]], np.zeros((3, 3)), 1, 3, [[0, 4, 8, 6, 8, 10, 6, 7], [32, 36, 40, 22, 24, 26, 14, 15], [64, 68, 72, 38, 40, 42, 22, 23]]), (np.arange(6), [0.25, 0.5], [-1, -2], 0, 3, [-1, 3, 7, 4, 6, 8]), (np.ones((9, 12)), np.ones((3, 4)), np.zeros((3, 4)), 0, 3, None), (np.ones((3, 4, 5, 6)), np.ones((3, 4)), np.zeros((3, 4)), 2, 2, None)])
    def test_blocked_quantize_linear(self, x, scale, zero_point, axis, block_size, expected):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.INT8, [None])
        scale_data = np.array(scale, dtype=np.float32)
        zp_data = np.array(zero_point, dtype=np.int8)
        model = make_model(make_graph([make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['Y'], axis=axis, block_size=block_size)], 'g', [X], [Y], [make_tensor('scale', TensorProto.FLOAT, scale_data.shape, scale_data), make_tensor('zero', TensorProto.INT8, scale_data.shape, zp_data)]))
        ref = ReferenceEvaluator(model)
        data = np.array(x, dtype=np.float32)
        if expected is not None:
            expected = np.array(expected, dtype=np.int8)
            got = ref.run(None, {'X': data})
            assert_allclose(expected, got[0])
        else:
            with self.assertRaises(ValueError):
                ref.run(None, {'X': data})

    @parameterized.parameterized.expand([(np.arange(12).reshape(3, 4), np.arange(1, 7).reshape(3, 2), np.zeros((3, 2)), 1, 2, [[0, 1, 4, 6], [12, 15, 24, 28], [40, 45, 60, 66]]), (np.arange(12).reshape(3, 4), np.arange(1, 7).reshape(3, 2), np.ones((3, 2)), 1, 2, [[-1, 0, 2, 4], [9, 12, 20, 24], [35, 40, 54, 60]]), (np.dstack([np.arange(4).reshape(2, 2)] * 4), np.dstack([np.array([[1, 1], [2, 3]]), np.array([[4, 5], [6, 7]])]), np.zeros((2, 2, 2)), 2, 2, [[[0, 0, 0, 0], [1, 1, 5, 5]], [[4, 4, 12, 12], [9, 9, 21, 21]]]), (np.arange(24).reshape(3, 8), [[2, 1, 3], [2, 1, 3], [2, 1, 3]], np.zeros((3, 3)), 1, 3, [[0, 2, 4, 3, 4, 5, 18, 21], [16, 18, 20, 11, 12, 13, 42, 45], [32, 34, 36, 19, 20, 21, 66, 69]]), (np.arange(6), [2, 3], [1, 2], 0, 3, [-2, 0, 2, 3, 6, 9]), (np.ones((9, 12)), np.ones((3, 4)), np.zeros((3, 4)), 0, 3, None), (np.ones((3, 4, 5, 6)), np.ones((3, 4)), np.zeros((3, 4)), 2, 2, None)])
    def test_blocked_dequantize_linear(self, x, scale, zero_point, axis, block_size, expected):
        X = make_tensor_value_info('X', TensorProto.INT8, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        scale_data = np.array(scale, dtype=np.float32)
        zp_data = np.array(zero_point, dtype=np.int8)
        model = make_model(make_graph([make_node('DequantizeLinear', ['X', 'scale', 'zero'], ['Y'], axis=axis, block_size=block_size)], 'g', [X], [Y], [make_tensor('scale', TensorProto.FLOAT, scale_data.shape, scale_data), make_tensor('zero', TensorProto.INT8, scale_data.shape, zp_data)]))
        ref = ReferenceEvaluator(model)
        data = np.array(x, dtype=np.int8)
        if expected is not None:
            expected = np.array(expected, dtype=np.float32)
            got = ref.run(None, {'X': data})
            assert_allclose(expected, got[0])
        else:
            with self.assertRaises(ValueError):
                ref.run(None, {'X': data})

    def test_lrn(self):

        def _expected(x, alpha, beta, bias, size):
            square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
            for n, c, h, w in np.ndindex(x.shape):
                square_sum[n, c, h, w] = sum(x[n, max(0, c - int(math.floor((size - 1) / 2))):min(5, c + int(math.ceil((size - 1) / 2)) + 1), h, w] ** 2)
            y = x / (bias + alpha / size * square_sum) ** beta
            return y
        alpha = 0.0002
        beta = 0.5
        bias = 2.0
        size = 3
        X = make_tensor_value_info('X', TensorProto.FLOAT, [5, 5, 50, 50])
        Z = make_tensor_value_info('Z', TensorProto.UNDEFINED, None)
        nodes = [make_node('LRN', ['X'], ['Z'], alpha=alpha, beta=beta, bias=bias, size=size)]
        model = make_model(make_graph(nodes, 'g', [X], [Z]))
        ref = ReferenceEvaluator(model)
        data = np.random.rand(5, 5, 5, 5).astype(np.float32)
        got = ref.run(None, {'X': data})
        expected = _expected(data, alpha, beta, bias, size)
        self.assertEqual(len(expected), len(got[0]))

    def test_conv_im2col_1d(self):
        feeds = {'X': np.arange(1 * 1 * 11).reshape((1, 1, 11)).astype(np.float32) + 1, 'W': np.arange(3).reshape((1, 1, 3)).astype(np.float32), 'B': np.zeros((1,), dtype=np.float32)}
        kwargs = dict(group=1, dilations=[1], kernel_shape=[3], pads=[1, 1], strides=[1], auto_pad='NOTSET')
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_1d_pad0(self):
        feeds = {'X': np.arange(2 * 4 * 3).reshape((2, 4, -1)).astype(np.float32) + 1, 'W': np.arange(2 * 4 * 3).reshape((-1, 4, 3)).astype(np.float32), 'B': np.zeros((1,), dtype=np.float32)}
        kwargs = dict(group=1, dilations=[1], kernel_shape=[3], pads=[0, 0], strides=[1], auto_pad='NOTSET')
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d(self):
        feeds = {'X': np.arange(1 * 1 * 11 * 23).reshape((1, 1, 11, 23)).astype(np.float32) + 1, 'W': np.arange(9).reshape((1, 1, 3, 3)).astype(np.float32), 'B': np.zeros((1,), dtype=np.float32)}
        kwargs = dict(group=1, dilations=[1, 1], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], auto_pad='NOTSET')
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d_pad0(self):
        feeds = {'X': np.arange(2 * 3 * 5 * 2).reshape((2, 3, 5, -1)).astype(np.float32) + 1, 'W': 2 ** np.arange(3 * 3 * 1 * 2).reshape((-1, 3, 1, 2)).astype(np.float32), 'B': np.zeros((1,), dtype=np.float32)}
        kwargs = dict(group=1, dilations=[1, 1], kernel_shape=[1, 2], pads=[0, 0, 0, 0], strides=[1, 1], auto_pad='NOTSET')
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d_autopad(self):
        feeds = {'X': np.arange(5 * 5).reshape((1, 1, 5, -1)).astype(np.float32) + 1, 'W': 2 ** np.arange(3 * 3).reshape((1, 1, 3, 3)).astype(np.float32), 'B': np.zeros((1,), dtype=np.float32)}
        kwargs = dict(group=1, dilations=[1, 1], kernel_shape=[3, 3], strides=[2, 2], pads=None, auto_pad='SAME_LOWER')
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_3d(self):
        feeds = {'X': np.arange(1 * 1 * 11 * 5 * 13).reshape((1, 1, 11, 5, 13)).astype(np.float32) + 1, 'W': np.arange(27).reshape((1, 1, 3, 3, 3)).astype(np.float32), 'B': np.zeros((1,), dtype=np.float32)}
        kwargs = dict(group=1, dilations=[1, 1, 1], kernel_shape=[3, 3, 3], pads=[1, 1, 1, 1, 1, 1], strides=[1, 1, 1], auto_pad='NOTSET')
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d_strides(self):
        feeds = {'X': np.arange(1 * 3 * 6 * 6).reshape((1, 3, 6, 6)).astype(np.float32) + 1, 'W': np.arange(2 * 3 * 3 * 3).reshape((2, 3, 3, 3)).astype(np.float32), 'B': np.zeros((2,), dtype=np.float32)}
        kwargs = dict(group=1, dilations=[1, 1], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2], auto_pad='NOTSET')
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    def test_conv_im2col_2d_dilations(self):
        feeds = {'X': np.arange(1 * 3 * 6 * 6).reshape((1, 3, 6, 6)).astype(np.float32) + 1, 'W': np.arange(2 * 3 * 3 * 3).reshape((2, 3, 3, 3)).astype(np.float32), 'B': np.zeros((2,), dtype=np.float32)}
        kwargs = dict(group=1, dilations=[2, 1], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2], auto_pad='NOTSET')
        expected = _conv_implementation(**feeds, **kwargs)
        got = _conv_implementation_im2col(**feeds, **kwargs)
        assert_allclose(expected, got)

    @parameterized.parameterized.expand([('ReduceSum',), ('ReduceL1',), ('ReduceL2',), ('ReduceMin',), ('ReduceMax',), ('ReduceProd',), ('ReduceSumSquare',)])
    def test_reduce_op_no_axis(self, op):
        X = make_tensor_value_info('X', TensorProto.FLOAT, None)
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
        data = np.arange(6).reshape((1, 3, 2)).astype(np.float32)
        nodes = [make_node(op, ['X'], ['Y'], keepdims=0)]
        model = make_model(make_graph(nodes, 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        got = ref.run(None, {'X': data})
        r = got[0]
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(r.shape, ())

    @parameterized.parameterized.expand([(1,), (2,), (3,), (4,), (5,), (6,)])
    def test_pad(self, dim):
        X = make_tensor_value_info('X', TensorProto.FLOAT, None)
        P = make_tensor_value_info('P', TensorProto.INT64, None)
        V = make_tensor_value_info('V', TensorProto.FLOAT, None)
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
        value = np.array([-5], dtype=np.float32)
        node = make_node('Pad', inputs=['X', 'P', 'V'], outputs=['Y'], mode='constant')
        model = make_model(make_graph([node], 'g', [X, P, V], [Y]))
        ref = ReferenceEvaluator(model)
        x = np.array([1], dtype=np.float32).reshape((1,) * dim)
        p = np.array([1, 1] * dim, dtype=np.int64)
        got = ref.run(None, {'X': x, 'P': p, 'V': value})[0]
        self.assertEqual(got.shape, (3,) * dim)
        self.assertEqual(got.dtype, np.float32)
        p = np.repeat([7, 3], dim).astype(np.int64)
        got = ref.run(None, {'X': x, 'P': p, 'V': value})[0]
        self.assertEqual(got.shape, (11,) * dim)
        self.assertEqual(got.dtype, np.float32)

    def test_constant_of_shape(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, None)
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
        nodes = [make_node('Shape', inputs=['X'], outputs=['shape']), make_node('ConstantOfShape', inputs=['shape'], outputs=['Y'], value=make_tensor('value', TensorProto.UINT16, [1], [1]))]
        model = make_model(make_graph(nodes, 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        x = np.array(1, dtype=np.float32)
        got = ref.run(None, {'X': x})[0]
        self.assertEqual(got.shape, tuple())
        self.assertEqual(got.dtype, np.uint16)
        assert_allclose(np.array(1, dtype=np.uint16), got)

    def test_constant_of_shape_castlike(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, None)
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, None)
        nodes = [make_node('Constant', [], ['like'], value=make_tensor('c', TensorProto.UINT16, [1], [2])), make_node('Shape', inputs=['X'], outputs=['shape']), make_node('ConstantOfShape', inputs=['shape'], outputs=['cst'], value=make_tensor('value', TensorProto.INT64, [1], [1])), make_node('CastLike', ['cst', 'like'], ['Y'])]
        model = make_model(make_graph(nodes, 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        x = np.array(1, dtype=np.float32)
        got = ref.run(None, {'X': x})[0]
        self.assertEqual(got.shape, tuple())
        self.assertEqual(got.dtype, np.uint16)
        assert_allclose(np.array(1, dtype=np.uint16), got)

    def test_dynamic_quantize_linear(self):
        feeds = {'X': np.array([[-0.0780749545, -0.380597055, 0.133831516, -0.0820474699, 0.0756645501, 0.0565112457, 0.256818235, 0.0942316353, 0.188027292, 0.144878656, 0.134825557, -0.20457691, 0.168852255, 0.062325336, 0.430482924, -0.0550433956, 0.0910681635, 0.155332625, -0.0453630984, 0.399910688, -0.128678545, 0.0377916731, 0.12987271, -0.112420328, -0.0297306702, 0.220508516, -0.00588933006, 0.481076002, -0.118835129, -0.0445004404, -0.0753675848, 0.14111267, 0.197793499, -0.771476865, 0.0864694864, 0.0173293594, 0.128247693, 0.075814411, -0.27143538, 0.175212905, -0.247283235, -0.0302810557, 0.0845039487, 0.602229357, -0.104913145, -0.246705681, 0.29207328, -0.0388464853, 0.426557302, -0.371325493, -0.311283618, 0.0785303488, 0.318069518, -0.151467413, -0.102828763, 0.092913188, 0.255233884, 0.500160515, -0.149993747, 0.429408073, -0.191787735, 0.0316187665, -0.0184284449, -0.162873864, -0.273632705, 0.284725696, -0.287029266, -0.0715534389, 0.224836454, -0.170527741, -0.265601039, -0.00268008932, 0.144260898, 0.0780707747, 0.0273875445, -0.118391573, -0.064497225, -0.00522445887, -0.296754301, 0.105800219, 0.262558222, 0.0362841524, 0.00944730639, 0.175837606, 0.269956529, 0.302758247, -0.113724738, 0.298936248, 0.0854668319, -0.6745556, 0.438643873, 0.0127896462, 0.0920789093, 0.193946883, 0.197548166, 0.282558739, -0.24887912, -0.393428057, 0.0645540953, -0.00966283306], [-0.242438495, -0.358334243, 0.122619808, -0.121529415, -0.0523974374, -0.0674922541, 0.109727375, -0.00356835872, 0.151029706, 0.118043356, 0.0816475376, -0.236466587, 0.0944180191, 0.0561679937, 0.367988586, 0.129441261, 0.115772486, 0.0530351102, -0.104345076, 0.129062623, -0.217205048, 0.025808949, 0.218848974, -0.0836039707, 0.00543577969, 0.078707628, 0.119966723, 0.281631678, -0.0358020402, -0.0965647772, 0.0317915753, 0.249396205, 0.22360079, -0.382718384, 0.116506606, -0.0319400802, 0.160812005, 0.098173596, -0.199046463, 0.0881427377, -0.165171087, -0.110251531, 0.115387037, 0.31931293, -0.114400804, -0.202447772, 0.233669549, 0.0920853689, 0.391551465, -0.358036369, -0.235071778, -0.0500670634, 0.165313914, -0.160922945, -0.195520848, 0.182456985, 0.380704433, 0.419890225, -0.0298131555, 0.366110623, -0.281170905, -0.12394245, 0.0958625227, -0.0590450205, -0.156236291, 0.211865619, -0.175286919, -0.0836539492, 0.129381597, -0.16611588, -0.166922957, 0.165688396, 0.0841224194, 0.167839468, -0.0403967649, -0.0834071711, -0.0565301552, 0.16707401, -0.319734544, 0.0771123618, 0.0557036921, 0.12070933, -0.0663790107, 0.136002287, 0.342324018, 0.342968374, -0.142380476, 0.289662749, 0.0182179566, -0.496056974, 0.264364302, 0.073891893, 0.111150607, -0.0595749579, 0.118562862, 0.0690007359, -0.208283514, -0.170682222, 0.0782715827, 0.135489792], [-0.0635757595, -0.320629895, 0.0938569903, -0.115190029, 0.010364607, -0.0431734361, 0.231717676, 0.0401005745, 0.118915148, 0.210071519, -0.0299234912, -0.293135524, -0.23958829, 0.0671441257, 0.315238893, -0.0508778207, 0.11614728, -0.0672954097, -0.163787514, 0.160288364, -0.230847582, -0.222037435, 0.0450191796, -0.109636143, -0.0600997508, 0.214693844, -0.00851289369, 0.388416052, -0.118085533, -0.0857385695, -0.0118666515, 0.329741478, 0.203779504, -0.169492334, 0.194324687, 0.0416374728, 0.0683876276, -0.0185160581, -0.0373600274, 0.0714804307, -0.301446438, 0.0174035393, 0.335123807, 0.417102218, -0.158562332, -0.0254483074, 0.199573949, 0.0795029774, 0.482958555, -0.458213627, -0.26722917, 0.127542794, 0.0447799414, -0.168686539, -0.0892183557, 0.0979782715, 0.0277656261, 0.296871901, 0.0629860088, 0.177246213, -0.315523535, -0.174582571, 0.0125724282, -0.0454988703, -0.129154682, 0.213568255, -0.140891463, -0.266211092, 0.262144595, -0.142889306, -0.0667349845, 0.0163380653, -0.0192995071, 0.114811368, 0.143584609, -0.0265347548, 0.0932592154, 0.223325342, -0.187100068, 0.057119742, 0.271253467, 0.102890521, -0.00304941833, 0.110537663, 0.275728375, 0.211693868, -0.180009842, 0.0299300496, 0.177923322, -0.453491032, 0.194149211, 0.247100577, -0.00995091349, 0.199301094, 0.20656465, -0.0199648179, -0.18962945, 0.016168993, 0.104817756, 0.206400946], [-0.038625136, -0.307115853, 0.0374236181, -0.171237886, -0.0277359486, -0.0408068746, 0.0691853091, 0.00265322998, 0.123958819, 0.220951259, 0.0836500078, -0.31441319, 0.134745941, -0.0825274512, 0.336270213, -0.0249634441, -0.0106189884, 0.0718819201, -0.173392966, 0.298084319, -0.125626653, -0.0217043106, 0.287982523, -0.00341932476, -0.0689984411, -0.00714176893, 0.044954244, 0.516424477, -0.102622584, 0.0202640779, 0.0230106711, 0.193037808, 0.203393996, -0.434632808, 0.0574068353, 0.0966466218, -0.0719890296, 0.0423505977, -0.160067812, 0.0388947129, -0.332329512, 0.191072702, 0.0379437394, 0.43383947, -0.12923196, -0.146881789, 0.247269243, 0.0286379829, 0.292926908, -0.297341049, -0.340239167, 0.152589783, 0.0881168991, 0.0140633471, -0.0883188471, 0.24836731, 0.341982782, 0.318781316, -0.115148552, 0.25432542, -0.182771236, 0.053388983, 0.0212034155, -0.0978844613, -0.161611915, 0.354084134, -0.125332132, -0.20798941, 0.0235610008, -0.135993093, -0.197377697, -0.0121356212, 0.00786775351, 0.471337497, 0.00949376822, 0.0425345525, 0.11416205, 0.0627847165, -0.231957644, -0.0833211765, 0.202719584, -0.0464919358, 0.0757966787, 0.101521172, 0.316580981, 0.149488643, -0.120770879, 0.256563038, 0.166572407, -0.611343801, 0.209183827, 0.0666101649, 0.177328646, 0.177777156, 0.203266457, 0.137545317, -0.138004154, -0.257656008, 0.18392086, 0.0287696868], [0.000514136627, -0.288997203, -0.0334554128, -0.0680408552, -0.0461972654, 0.175428241, 0.186710209, -0.151355267, 0.128381401, 0.287129283, 0.0052215457, -0.353413224, -0.0387947261, 0.0581918843, 0.317016482, -0.251671404, 0.0701867491, -0.0492945537, -0.273323953, 0.127938241, -0.411552131, -0.0223789401, 0.195418939, -0.325946212, 0.0460528135, 0.18688409, 0.0698191971, 0.295638293, -0.180322871, -0.0298620313, 0.211789399, 0.31514591, 0.311763227, -0.578147054, 0.0643244758, -0.0314367823, 0.186190963, 0.171108633, -0.0629745722, 0.0748428777, -0.458003521, -0.3014718, 0.217973694, 0.573273778, -0.101379365, -0.246951967, 0.158989042, -0.179126799, 0.524153829, -0.46485284, -0.294867605, 0.183558539, 0.250552952, -0.0856962949, -0.25755471, 0.230709136, 0.353280157, 0.320184112, 0.0299184099, 0.309098989, -0.202217728, -0.0929201543, -0.0120569356, -0.137087986, -0.21652469, 0.139787242, -0.12790215, -0.264347821, 0.0929943919, -0.157217175, -0.386638314, 0.0790465698, 0.040721193, -0.0307695866, 0.127393469, -0.118648581, -0.0721216127, -0.0371141285, -0.337082207, -0.0623112544, 0.352166295, 0.151260465, 0.050342761, 0.190433189, 0.521304548, 0.385341585, -0.12645705, 0.154961571, 0.0529025272, -0.506486952, 0.228533953, 0.178438187, -0.041476503, 0.201903239, -0.0389365852, 0.0884043872, -0.155351788, -0.0906582028, 0.0995599255, -0.047998976]], dtype=np.float32)}
        expected = [np.array([[129, 72, 168, 128, 157, 153, 191, 160, 178, 170, 168, 105, 174, 155, 223, 133, 160, 172, 135, 217, 119, 150, 167, 122, 137, 184, 142, 232, 121, 135, 129, 169, 180, 0, 159, 146, 167, 157, 93, 176, 97, 137, 159, 255, 124, 97, 197, 136, 222, 74, 85, 158, 202, 115, 124, 160, 190, 236, 115, 223, 107, 149, 140, 113, 92, 196, 90, 130, 185, 111, 94, 143, 170, 157, 148, 121, 131, 142, 88, 163, 192, 150, 145, 176, 193, 199, 122, 198, 159, 18, 224, 145, 160, 179, 180, 195, 97, 70, 155, 141], [98, 76, 166, 120, 133, 130, 163, 142, 171, 165, 158, 99, 161, 153, 211, 167, 164, 153, 124, 167, 103, 148, 184, 127, 144, 158, 165, 195, 136, 125, 149, 189, 185, 72, 165, 137, 173, 161, 106, 159, 112, 123, 164, 202, 122, 105, 186, 160, 216, 77, 99, 134, 174, 113, 107, 177, 214, 221, 137, 211, 91, 120, 161, 132, 114, 182, 110, 127, 167, 112, 112, 174, 159, 174, 136, 128, 133, 174, 84, 157, 153, 165, 131, 168, 207, 207, 117, 197, 146, 51, 192, 157, 164, 132, 165, 156, 104, 111, 158, 168], [131, 83, 160, 122, 145, 135, 186, 150, 165, 182, 137, 89, 99, 155, 202, 134, 165, 131, 113, 173, 100, 102, 151, 123, 132, 183, 141, 215, 121, 127, 141, 204, 181, 112, 179, 151, 156, 140, 136, 156, 87, 146, 205, 220, 114, 138, 180, 158, 233, 58, 93, 167, 151, 112, 126, 161, 148, 198, 155, 176, 84, 111, 145, 135, 119, 183, 117, 94, 192, 116, 131, 146, 139, 164, 170, 138, 160, 184, 108, 154, 193, 162, 142, 164, 194, 182, 110, 149, 176, 59, 179, 189, 141, 180, 181, 139, 108, 146, 162, 181], [136, 86, 150, 111, 138, 135, 156, 143, 166, 184, 159, 85, 168, 128, 205, 138, 141, 156, 111, 198, 120, 139, 196, 142, 130, 142, 151, 239, 124, 147, 147, 179, 181, 62, 154, 161, 130, 151, 113, 150, 81, 178, 150, 224, 119, 116, 189, 148, 197, 88, 80, 171, 159, 146, 127, 189, 206, 202, 122, 190, 109, 153, 147, 125, 113, 209, 120, 104, 147, 118, 106, 141, 144, 230, 145, 151, 164, 155, 100, 128, 181, 134, 157, 162, 202, 171, 121, 191, 174, 30, 182, 155, 176, 176, 181, 169, 117, 95, 177, 148], [143, 89, 137, 130, 134, 176, 178, 115, 167, 196, 144, 77, 136, 154, 202, 96, 156, 134, 92, 167, 67, 139, 179, 82, 152, 178, 156, 198, 110, 137, 182, 202, 201, 36, 155, 137, 178, 175, 131, 157, 58, 87, 183, 249, 124, 97, 173, 110, 240, 57, 88, 177, 190, 127, 95, 186, 209, 202, 149, 200, 105, 126, 141, 118, 103, 169, 119, 94, 160, 114, 71, 158, 151, 137, 167, 121, 130, 136, 80, 131, 208, 171, 152, 178, 240, 215, 120, 172, 153, 49, 185, 176, 135, 180, 136, 159, 114, 126, 161, 134]], dtype=np.uint8), np.array(0.005387083161622286, dtype=np.float32), np.array(143, dtype=np.uint8)]
        X = make_tensor_value_info('X', TensorProto.FLOAT, None)
        Y = make_tensor_value_info('Y', TensorProto.UINT8, None)
        Scale = make_tensor_value_info('scale', TensorProto.FLOAT, None)
        Zp = make_tensor_value_info('zp', TensorProto.UINT8, None)
        nodes = [make_node('DynamicQuantizeLinear', ['X'], ['Y', 'scale', 'zp'])]
        model = make_model(make_graph(nodes, 'g', [X], [Y, Scale, Zp]), opset_imports=[make_opsetid('', onnx_opset_version() - 1)])
        ref = ReferenceEvaluator(model)
        got = ref.run(None, feeds)
        self.assertEqual(len(got), 3)
        for i in range(2, -1, -1):
            assert_allclose(expected[i], got[i])

    @parameterized.parameterized.expand([(['abc', 'def'], ['.com', '.net'], ['abc.com', 'def.net'], (2,)), (['cat', 'dog', 'snake'], ['s'], ['cats', 'dogs', 'snakes'], (3,)), ('cat', 's', 'cats', ()), (['a', 'ß', 'y'], ['a', 'ß', 'y'], ['aa', 'ßß', 'yy'], (3,))])
    def test_string_concat(self, a, b, expected, expected_shape):
        A = make_tensor_value_info('A', TensorProto.STRING, None)
        B = make_tensor_value_info('B', TensorProto.STRING, None)
        Y = make_tensor_value_info('Y', TensorProto.STRING, None)
        node = make_node('StringConcat', inputs=['A', 'B'], outputs=['Y'])
        model = make_model(make_graph([node], 'g', [A, B], [Y]))
        ref = ReferenceEvaluator(model)
        result, *_ = ref.run(None, {'A': np.array(a), 'B': np.array(b)})
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype.kind, 'O')
        self.assertEqual(result.shape, expected_shape)

    @parameterized.parameterized.expand([(['1,2,3', '4,5,6'], ',', None, [['1', '2', '3'], ['4', '5', '6']], [3, 3]), (['1,', '4,6', ''], ',', None, [['1', ''], ['4', '6'], ['', '']], [2, 2, 1]), (['1', '4,6', '4,5,6'], ',', 1, [['1', ''], ['4', '6'], ['4', '5,6']], [1, 2, 2]), ([['1,', '4,6', '4,5,6'], ['1,', '4,6', '4,5,6']], ',', None, [[['1', '', ''], ['4', '6', ''], ['4', '5', '6']], [['1', '', ''], ['4', '6', ''], ['4', '5', '6']]], [[2, 2, 3], [2, 2, 3]]), (['hello world !', '  hello   world !', ' hello world   ! '], None, None, [['hello', 'world', '!'], ['hello', 'world', '!'], ['hello', 'world', '!']], [3, 3, 3]), (['hello world !', '  hello   world !', ' hello world   ! '], '', None, [['hello', 'world', '!'], ['hello', 'world', '!'], ['hello', 'world', '!']], [3, 3, 3]), (['o-n-n--x-', 'o-n----nx'], '-', None, [['o', 'n', 'n', '', 'x', ''], ['o', 'n', '', '', '', 'nx']], [6, 6]), ([], ' ', 2, np.array([]).reshape((0, 0)), [])])
    def test_string_split(self, x, delimiter, maxsplit, expected_split, expected_num_splits):
        X = make_tensor_value_info('X', TensorProto.STRING, None)
        Splits = make_tensor_value_info('Splits', TensorProto.STRING, None)
        MaxSplits = make_tensor_value_info('MaxSplits', TensorProto.INT32, None)
        node = make_node('StringSplit', inputs=['X'], outputs=['Splits', 'MaxSplits'], delimiter=delimiter, maxsplit=maxsplit)
        model = make_model(make_graph([node], 'g', [X], [Splits, MaxSplits]))
        ref = ReferenceEvaluator(model)
        x = np.array(x, dtype=object)
        result, num_splits, *_ = ref.run(None, {'X': x})
        np.testing.assert_array_equal(result, np.array(expected_split, dtype=object))
        np.testing.assert_array_equal(num_splits, np.array(expected_num_splits, dtype=np.int64))

    def test_qlinearconv_int8(self):
        node = make_node('QLinearMatMul', inputs=['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'], outputs=['y'])
        graph = make_graph([node], 'g', [make_tensor_value_info('a', TensorProto.FLOAT, [None, None]), make_tensor_value_info('a_scale', TensorProto.FLOAT, [1]), make_tensor_value_info('a_zero_point', TensorProto.INT8, [1]), make_tensor_value_info('b', TensorProto.FLOAT, [None, None]), make_tensor_value_info('b_scale', TensorProto.FLOAT, [1]), make_tensor_value_info('b_zero_point', TensorProto.INT8, [1]), make_tensor_value_info('y_scale', TensorProto.FLOAT, [1]), make_tensor_value_info('y_zero_point', TensorProto.INT8, [1])], [make_tensor_value_info('y', TensorProto.FLOAT, [None, None])])
        onnx_model = make_model(graph, opset_imports=[make_opsetid('', 20)], ir_version=9)
        sess = ReferenceEvaluator(onnx_model)
        a = np.array([[208, 236, 0, 238], [3, 214, 255, 29]])
        a -= 127
        a = a.astype(np.int8)
        a_scale = np.array([0.0066], dtype=np.float32)
        a_zero_point = np.array([113 - 127], dtype=np.int8)
        b = np.array([[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]])
        b -= 127
        b = b.astype(np.int8)
        b_scale = np.array([0.00705], dtype=np.float32)
        b_zero_point = np.array([114 - 127], dtype=np.int8)
        y_scale = np.array([0.0107], dtype=np.float32)
        y_zero_point = np.array([118 - 127], dtype=np.int8)
        got = sess.run(None, dict(a=a, a_scale=a_scale, a_zero_point=a_zero_point, b=b, b_scale=b_scale, b_zero_point=b_zero_point, y_scale=y_scale, y_zero_point=y_zero_point))
        np.testing.assert_array_equal(np.array([[41, -12, -9], [1, -75, 20]], dtype=np.int8), got[0])

    @parameterized.parameterized.expand([(['www.google.com', 'www.facebook.com', 'www.bbc.co.uk'], 'www\\.[\\w.-]+\\.\\bcom\\b', [True, True, False], (3,)), ([['Onnx', 'tensorflow', 'Numpy'], ['Pytorch', 'Cython', 'numba']], '^[A-Z][a-z]*$', [[True, False, True], [True, True, False]], (2, 3)), (['account@gmail.com', 'account@hotmail.com', 'not email', 'account2@yahoo.com'], '(\\W|^)[\\w.\\-]{0,25}@(yahoo|gmail)\\.com(\\W|$)', [True, False, False, True], (4,))])
    @unittest.skipIf(sys.platform == 'win32', 'google-re2 package is not built for win32')
    def test_regex_full_match(self, x, pattern, expected, expected_shape):
        X = make_tensor_value_info('X', TensorProto.STRING, None)
        Y = make_tensor_value_info('Y', TensorProto.BOOL, None)
        node = make_node('RegexFullMatch', inputs=['X'], outputs=['Y'], pattern=pattern)
        model = make_model(make_graph([node], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        result, *_ = ref.run(None, {'X': np.array(x)})
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype.kind, 'b')
        self.assertEqual(result.shape, expected_shape)

    @unittest.skipIf(sys.platform == 'win32', 'google-re2 package is not built for win32')
    def test_regex_invalid_pattern(self):
        X = make_tensor_value_info('X', TensorProto.STRING, None)
        Y = make_tensor_value_info('Y', TensorProto.BOOL, None)
        node = make_node('RegexFullMatch', inputs=['X'], outputs=['Y'], pattern='x)')
        model = make_model(make_graph([node], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        with self.assertRaises(ValueError):
            ref.run(None, {'X': np.array(['x'])})

    @parameterized.parameterized.expand([(TensorProto.UINT4, [-1, 0, 1.5, 2, 3.3, 10, 20, 40], [0, 0, 2, 2, 4, 10, 20, 30]), (TensorProto.UINT4, [-1, 0, 1.5, 2, 3.3, 10, 40], [0, 0, 2, 2, 4, 10, 30]), (TensorProto.UINT4, [0], [0]), (TensorProto.INT4, [-20, -14.5, 0, 1.5, 2, 3.3, 10, 20], [-16, -14, 0, 2, 2, 4, 10, 14]), (TensorProto.INT4, [-20, -14.5, 0, 1.5, 2, 3.3, 10], [-16, -14, 0, 2, 2, 4, 10]), (TensorProto.INT4, [0], [0])])
    @unittest.skipIf(version_utils.numpy_older_than('1.22.0'), 'The test requires numpy 1.22.0 or later')
    def test_quantize_linear_int4(self, qtype, data, expected):
        X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
        Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
        model = make_model(make_graph([make_node('Constant', [], ['scale'], value=make_tensor('scale', TensorProto.FLOAT, [1], [2.0])), make_node('Constant', [], ['zero'], value=make_tensor('zero', qtype, [1], [0])), make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['T']), make_node('DequantizeLinear', ['T', 'scale'], ['Y'], axis=0)], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        got = ref.run(None, {'X': np.asarray(data)})
        assert_allclose(expected, got[0])

    @parameterized.parameterized.expand(itertools.product((TensorProto.FLOAT, TensorProto.FLOAT16), (TensorProto.UINT4, TensorProto.INT4)))
    def test_cast_int4_output(self, cast_from, cast_to):
        X = make_tensor_value_info('X', cast_from, [None])
        Y = make_tensor_value_info('Y', cast_to, [None])
        model = make_model(make_graph([make_node('Cast', ['X'], ['Y'], to=cast_to)], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        data = np.array([0, 1, 2.4, 2.6, 4, 10], dtype=np.float32)
        signed = cast_to == TensorProto.INT4
        expected1 = np.array([subbyte.float32_to_4bit_unpacked(x, signed=signed) for x in data])
        got = ref.run(None, {'X': data})
        self.assertEqual(expected1.tolist(), got[0].tolist())

    @parameterized.parameterized.expand(itertools.product((TensorProto.UINT4, TensorProto.INT4), (TensorProto.FLOAT, TensorProto.FLOAT16)))
    def test_cast_int4_input(self, cast_from, cast_to):
        X = make_tensor_value_info('X', cast_from, [None])
        Y = make_tensor_value_info('Y', cast_to, [None])
        model = make_model(make_graph([make_node('Cast', ['X'], ['Y'], to=TensorProto.FLOAT)], 'g', [X], [Y]))
        ref = ReferenceEvaluator(model)
        data = np.array(range(0, 7), dtype=np.float32)
        cast_from_np = custom.uint4 if cast_from == TensorProto.UINT4 else custom.int4
        data = data.astype(cast_from_np)
        expected1 = np.array([subbyte.float32_to_4bit_unpacked(x, cast_from_np) for x in data])
        got = ref.run(None, {'X': data})
        self.assertEqual(expected1.tolist(), got[0].tolist())

    def test_a_function_calling_a_function_once(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, ['N'])
        output = make_tensor_value_info('output', TensorProto.FLOAT, ['N'])
        Z = make_tensor_value_info('output', TensorProto.FLOAT, ['N'])
        func_def_add = make_function('this', 'fctadd', ['input2'], ['output'], [make_node('Constant', [], ['one'], value_floats=[1.0], name='CC0'), make_node('Add', ['input2', 'one'], ['output'], name='A1')], opset_imports=[make_operatorsetid('', 15)])
        func_def = make_function('this', 'fct', ['input'], ['output'], [make_node('Constant', [], ['one'], value_floats=[1.0], name='CC'), make_node('Greater', ['input', 'one'], ['cond']), make_node('If', ['cond'], ['output'], then_branch=make_graph([make_node('fctadd', ['input'], ['output'], domain='this')], 'gthen', [], [output]), else_branch=make_graph([make_node('Add', ['input', 'one'], ['output'], domain='')], 'gelse', [], [output]), name=':IF')], opset_imports=[make_operatorsetid('', 15), make_operatorsetid('this', 1)])
        model_def = make_model(make_graph([make_node('fct', ['X'], ['output'], domain='this')], 'test', [X], [Z]), ir_version=7, opset_imports=[make_operatorsetid('', 15), make_operatorsetid('this', 1)], functions=[func_def_add, func_def])
        feeds = {'X': np.array([-5], dtype=np.float32)}
        oinf = ReferenceEvaluator(model_def)
        expected = oinf.run(None, feeds)
        self.assertEqual(expected[0], np.array([-4], dtype=np.float32))

    def test_a_function_calling_a_function_double(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, ['N'])
        output = make_tensor_value_info('output', TensorProto.FLOAT, ['N'])
        Z = make_tensor_value_info('output', TensorProto.FLOAT, ['N'])
        func_def_add = make_function('this', 'fctadd', ['input2'], ['output'], [make_node('Constant', [], ['one'], value_floats=[1.0], name='CC0'), make_node('Add', ['input2', 'one'], ['output'], name='A1')], opset_imports=[make_operatorsetid('', 15)])
        func_def = make_function('this', 'fct', ['input'], ['output'], [make_node('Constant', [], ['one'], value_floats=[1.0], name='CC'), make_node('Greater', ['input', 'one'], ['cond']), make_node('If', ['cond'], ['output'], then_branch=make_graph([make_node('fctadd', ['input'], ['output'], domain='this')], 'gthen', [], [output]), else_branch=make_graph([make_node('Add', ['input', 'one'], ['output'], domain='')], 'gelse', [], [output]), name=':IF')], opset_imports=[make_operatorsetid('', 15), make_operatorsetid('this', 1)])
        model_def = make_model(make_graph([make_node('fct', ['X'], ['ztmp'], domain='this'), make_node('fct', ['ztmp'], ['output'], domain='this')], 'test', [X], [Z]), ir_version=7, opset_imports=[make_operatorsetid('', 15), make_operatorsetid('this', 1)], functions=[func_def_add, func_def])
        feeds = {'X': np.array([-5], dtype=np.float32)}
        oinf = ReferenceEvaluator(model_def)
        expected = oinf.run(None, feeds)
        self.assertEqual(expected[0], np.array([-3], dtype=np.float32))

    def test_overload_reference_implementation(self):
        X = make_tensor_value_info('X', TensorProto.FLOAT, ['N'])
        output = make_tensor_value_info('output', TensorProto.FLOAT, ['N'])
        Z = make_tensor_value_info('output', TensorProto.FLOAT, ['N'])
        func_def_add = make_function('this', 'fctadd', ['input2'], ['output'], [make_node('Constant', [], ['one'], value_floats=[1.0], name='CC0'), make_node('Add', ['input2', 'one'], ['output'], name='A1')], opset_imports=[make_operatorsetid('', 15)])
        func_def = make_function('this', 'fct', ['input'], ['output'], [make_node('Constant', [], ['one'], value_floats=[1.0], name='CC'), make_node('Greater', ['input', 'one'], ['cond']), make_node('If', ['cond'], ['output'], then_branch=make_graph([make_node('fctadd', ['input'], ['output'], domain='this')], 'gthen', [], [output]), else_branch=make_graph([make_node('Add', ['input', 'one'], ['output'], domain='')], 'gelse', [], [output]), name=':IF')], opset_imports=[make_operatorsetid('', 15), make_operatorsetid('this', 1)])
        model_def = make_model(make_graph([make_node('fct', ['X'], ['ztmp'], domain='this'), make_node('fct', ['ztmp'], ['output'], domain='this')], 'test', [X], [Z]), ir_version=7, opset_imports=[make_operatorsetid('', 15), make_operatorsetid('this', 1)], functions=[func_def_add, func_def])

        class MyReferenceEvaluator(ReferenceEvaluator):
            pass
        oinf = MyReferenceEvaluator(model_def)
        for v in oinf.functions_.values():
            self.assertIsInstance(v, MyReferenceEvaluator)