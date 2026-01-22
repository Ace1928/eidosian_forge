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
@staticmethod
def _get_test_tree_ensemble_classifier_binary(post_transform):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    In = make_tensor_value_info('I', TensorProto.INT64, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    node1 = make_node('TreeEnsembleClassifier', ['X'], ['I', 'Y'], domain='ai.onnx.ml', class_ids=[0, 0, 0, 0, 0, 0, 0], class_nodeids=[2, 3, 5, 6, 1, 3, 4], class_treeids=[0, 0, 0, 0, 1, 1, 1], class_weights=[0.0, 0.1764705926179886, 0.0, 0.5, 0.0, 0.0, 0.4285714328289032], classlabels_int64s=[0, 1], nodes_falsenodeids=[4, 3, 0, 0, 6, 0, 0, 2, 0, 4, 0, 0], nodes_featureids=[2, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0], nodes_hitrates=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], nodes_missing_value_tracks_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], nodes_modes=['BRANCH_LEQ', 'BRANCH_LEQ', 'LEAF', 'LEAF', 'BRANCH_LEQ', 'LEAF', 'LEAF', 'BRANCH_LEQ', 'LEAF', 'BRANCH_LEQ', 'LEAF', 'LEAF'], nodes_nodeids=[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4], nodes_treeids=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], nodes_truenodeids=[1, 2, 0, 0, 5, 0, 0, 1, 0, 3, 0, 0], nodes_values=[0.6874135732650757, -0.3654803931713104, 0.0, 0.0, -1.926770806312561, 0.0, 0.0, -0.3654803931713104, 0.0, -2.0783839225769043, 0.0, 0.0], post_transform=post_transform)
    graph = make_graph([node1], 'ml', [X], [In, Y])
    onx = make_model_gen_version(graph, opset_imports=[make_opsetid('', TARGET_OPSET), make_opsetid('ai.onnx.ml', 3)])
    check_model(onx)
    return onx