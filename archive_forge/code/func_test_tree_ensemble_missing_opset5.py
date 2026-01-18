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
@parameterized.expand([(input_type,) for input_type in [TensorProto.FLOAT, TensorProto.DOUBLE]])
def test_tree_ensemble_missing_opset5(self, input_type):
    model = self._get_test_tree_ensemble_opset_latest(AggregationFunction.SUM, Mode.LEQ, True, input_type)
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(input_type)
    x = np.arange(9).reshape((-1, 3)).astype(np_dtype) / 10 - 0.5
    x[2, 0] = 5
    x[1, :] = np.nan
    expected = np.array([[100001.0], [100100.0], [100100.0]], dtype=np_dtype)
    session = ReferenceEvaluator(model)
    actual, = session.run(None, {'X': x})
    assert_allclose(expected, actual, atol=1e-06)