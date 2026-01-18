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
def test_svm_classifier_linear(self):
    x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
    nan = np.nan
    expected_post = {'NONE': (np.array([2, 3, 0], dtype=np.int64), np.array([[-0.118086, -0.456685, 0.415783, 0.334506], [-0.061364, -0.231444, 0.073899, 0.091242], [-0.004642, -0.006203, -0.267985, -0.152023]], dtype=np.float32)), 'LOGISTIC': (np.array([2, 3, 0], dtype=np.int64), np.array([[0.470513, 0.387773, 0.602474, 0.582855], [0.484664, 0.442396, 0.518466, 0.522795], [0.498839, 0.498449, 0.433402, 0.462067]], dtype=np.float32)), 'SOFTMAX': (np.array([2, 3, 0], dtype=np.int64), np.array([[0.200374, 0.14282, 0.341741, 0.315065], [0.240772, 0.203115, 0.275645, 0.280467], [0.275491, 0.275061, 0.211709, 0.237739]], dtype=np.float32)), 'SOFTMAX_ZERO': (np.array([2, 3, 0], dtype=np.int64), np.array([[0.200374, 0.14282, 0.341741, 0.315065], [0.240772, 0.203115, 0.275645, 0.280467], [0.275491, 0.275061, 0.211709, 0.237739]], dtype=np.float32)), 'PROBIT': (np.array([2, 3, 0], dtype=np.int64), np.array([[nan, nan, -0.212698, -0.427529], [nan, nan, -1.447414, -1.333286], [nan, nan, nan, nan]], dtype=np.float32))}
    for post, expected in expected_post.items():
        with self.subTest(post_transform=post):
            onx = self._get_test_svm_classifier_binary(post, probability=True, linear=True)
            self._check_ort(onx, {'X': x}, rev=True, atol=1e-05)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, {'X': x})
            assert_allclose(expected[1], got[1], atol=1e-06)
            assert_allclose(expected[0], got[0])