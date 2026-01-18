import itertools
import unittest
from functools import wraps
from os import getenv
import numpy as np  # type: ignore
from numpy.testing import assert_allclose  # type: ignore
from parameterized import parameterized
import onnx
from onnx import ONNX_ML, TensorProto, TypeProto, ValueInfoProto
from onnx.checker import check_model
from onnx.defs import onnx_ml_opset_version, onnx_opset_version
from onnx.helper import (
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.aionnxml.op_tree_ensemble import (
@unittest.skipIf(not ONNX_ML, reason='onnx not compiled with ai.onnx.ml')
def test_svm_classifier_binary(self):
    x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
    expected_post = {'NONE': (np.array([0, 1, 1], dtype=np.int64), np.array([[0.993287, 0.006713], [0.469401, 0.530599], [0.014997, 0.985003]], dtype=np.float32)), 'LOGISTIC': (np.array([0, 1, 1], dtype=np.int64), np.array([[0.729737, 0.501678], [0.615242, 0.629623], [0.503749, 0.7281]], dtype=np.float32)), 'SOFTMAX': (np.array([0, 1, 1], dtype=np.int64), np.array([[0.728411, 0.271589], [0.484705, 0.515295], [0.274879, 0.725121]], dtype=np.float32)), 'SOFTMAX_ZERO': (np.array([0, 1, 1], dtype=np.int64), np.array([[0.728411, 0.271589], [0.484705, 0.515295], [0.274879, 0.725121]], dtype=np.float32)), 'PROBIT': (np.array([0, 1, 1], dtype=np.int64), np.array([[2.469393, -2.469391], [-0.076776, 0.076776], [-2.16853, 2.16853]], dtype=np.float32))}
    for post, expected in expected_post.items():
        with self.subTest(post_transform=post):
            onx = self._get_test_svm_classifier_binary(post)
            self._check_ort(onx, {'X': x}, rev=True, atol=1e-05)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, {'X': x})
            assert_allclose(expected[1], got[1], atol=1e-05)
            assert_allclose(expected[0], got[0])