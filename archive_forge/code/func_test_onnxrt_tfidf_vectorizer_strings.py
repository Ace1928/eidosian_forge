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
def test_onnxrt_tfidf_vectorizer_strings(self):
    inputi = np.array([['i1', 'i1', 'i3', 'i3', 'i3', 'i7'], ['i8', 'i6', 'i7', 'i5', 'i6', 'i8']])
    output = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]]).astype(np.float32)
    ngram_counts = np.array([0, 4]).astype(np.int64)
    ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
    pool_strings = np.array(['i2', 'i3', 'i5', 'i4', 'i5', 'i6', 'i7', 'i8', 'i6', 'i7'])
    model = make_model_gen_version(make_graph([make_node('TfIdfVectorizer', ['tokens'], ['out'], mode='TF', min_gram_length=2, max_gram_length=2, max_skip_count=0, ngram_counts=ngram_counts, ngram_indexes=ngram_indexes, pool_strings=pool_strings)], 'tfidf', [make_tensor_value_info('tokens', TensorProto.INT64, [None, None])], [make_tensor_value_info('out', TensorProto.FLOAT, [None, None])]), opset_imports=OPSETS)
    oinf = ReferenceEvaluator(model)
    res = oinf.run(None, {'tokens': inputi})
    self.assertEqual(output.tolist(), res[0].tolist())