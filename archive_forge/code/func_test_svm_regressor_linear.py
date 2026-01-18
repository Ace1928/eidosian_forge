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
def test_svm_regressor_linear(self):
    x = (np.arange(9).reshape((-1, 3)) - 5).astype(np.float32) / 5
    expected_post = {'NONE': (np.array([[0.96869], [1.132491], [1.296293]], dtype=np.float32),)}
    for post, expected in expected_post.items():
        with self.subTest(post_transform=post):
            onx = self._get_test_svm_regressor_linear(post)
            self._check_ort(onx, {'X': x}, rev=True, atol=1e-05)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, {'X': x})
            assert_allclose(expected[0], got[0], atol=1e-06)