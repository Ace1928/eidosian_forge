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
def test_svm_regressor(self):
    x = np.arange(9).reshape((-1, 3)).astype(np.float32) / 10 - 0.5
    expected_kernel = {'LINEAR': ([0.42438405752182007, 0.0, 3.0], np.array([[-0.468206], [0.227487], [0.92318]], dtype=np.float32)), 'POLY': ([0.3426632285118103, 0.0, 3.0], np.array([[0.527084], [0.543578], [0.546506]], dtype=np.float32)), 'RBF': ([0.30286383628845215, 0.0, 3.0], np.array([[0.295655], [0.477876], [0.695292]], dtype=np.float32)), 'SIGMOID': ([0.30682486295700073, 0.0, 3.0], np.array([[0.239304], [0.448929], [0.661689]], dtype=np.float32))}
    for kernel, (params, expected) in expected_kernel.items():
        with self.subTest(kernel=kernel):
            onx = self._get_test_svm_regressor(kernel, params)
            self._check_ort(onx, {'X': x}, atol=1e-06)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, {'X': x})
            assert_allclose(expected, got[0], atol=1e-06)