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
@parameterized.expand(tuple(itertools.chain.from_iterable((((AggregationFunction.SUM if opset5 else 'SUM', np.array([[0.576923], [0.576923], [0.576923]], dtype=np.float32), opset5), (AggregationFunction.AVERAGE if opset5 else 'AVERAGE', np.array([[0.288462], [0.288462], [0.288462]], dtype=np.float32), opset5), (AggregationFunction.MIN if opset5 else 'MIN', np.array([[0.076923], [0.076923], [0.076923]], dtype=np.float32), opset5), (AggregationFunction.MAX if opset5 else 'MAX', np.array([[0.5], [0.5], [0.5]], dtype=np.float32), opset5)) for opset5 in [True, False]))))
@unittest.skipIf(not ONNX_ML, reason='onnx not compiled with ai.onnx.ml')
def test_tree_ensemble_regressor_aggregation_functions(self, aggregate_function, expected_result, opset5):
    x = np.arange(9).reshape((-1, 3)).astype(np.float32) / 10 - 0.5
    model_factory = self._get_test_tree_ensemble_opset_latest if opset5 else self._get_test_tree_ensemble_regressor
    model_proto = model_factory(aggregate_function)
    sess = ReferenceEvaluator(model_proto)
    actual, = sess.run(None, {'X': x})
    assert_allclose(expected_result, actual, atol=1e-06)