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
def _get_test_svm_regressor_linear(post_transform, one_class=0):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    kwargs = {'coefficients': [0.28290501, -0.0266512, 0.01674867], 'kernel_params': [0.001, 0.0, 3.0], 'kernel_type': 'LINEAR', 'rho': [1.24032312], 'post_transform': post_transform, 'n_supports': 0, 'one_class': one_class}
    node1 = make_node('SVMRegressor', ['X'], ['Y'], domain='ai.onnx.ml', **kwargs)
    graph = make_graph([node1], 'ml', [X], [Y])
    onx = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(onx)
    return onx