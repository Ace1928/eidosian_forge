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
def test_single_tree_ensemble(self):
    X = make_tensor_value_info('X', TensorProto.DOUBLE, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.DOUBLE, [None, None])
    node = make_node('TreeEnsemble', ['X'], ['Y'], domain='ai.onnx.ml', n_targets=2, membership_values=None, nodes_missing_value_tracks_true=None, nodes_hitrates=None, aggregate_function=1, post_transform=PostTransform.NONE, tree_roots=[0], nodes_modes=make_tensor('nodes_modes', TensorProto.UINT8, (3,), [Mode.LEQ] * 3), nodes_featureids=[0, 0, 0], nodes_splits=make_tensor('nodes_splits', TensorProto.DOUBLE, (3,), np.array([3.14, 1.2, 4.2], dtype=np.float64)), nodes_truenodeids=[1, 0, 1], nodes_trueleafs=[0, 1, 1], nodes_falsenodeids=[2, 2, 3], nodes_falseleafs=[0, 1, 1], leaf_targetids=[0, 1, 0, 1], leaf_weights=make_tensor('leaf_weights', TensorProto.DOUBLE, (4,), np.array([5.23, 12.12, -12.23, 7.21], dtype=np.float64)))
    graph = make_graph([node], 'ml', [X], [Y])
    model = make_model_gen_version(graph, opset_imports=[make_opsetid('', TARGET_OPSET), make_opsetid('ai.onnx.ml', 5)])
    check_model(model)
    session = ReferenceEvaluator(model)
    output, = session.run(None, {'X': np.array([1.2, 3.4, -0.12, 1.66, 4.14, 1.77], np.float64).reshape(3, 2)})
    np.testing.assert_equal(output, np.array([[5.23, 0], [5.23, 0], [0, 12.12]], dtype=np.float64))