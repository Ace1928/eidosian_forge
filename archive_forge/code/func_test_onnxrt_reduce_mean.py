import itertools
import math
import sys
import unittest
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from os import getenv
from textwrap import dedent
from typing import Sequence, Tuple
import numpy as np
import parameterized
import version_utils
from numpy.testing import assert_allclose
import onnx.reference.custom_element_types as custom
from onnx import (
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32, from_array
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun, OpRunExpand
from onnx.reference.ops import load_op
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
from onnx.reference.ops._op_list import Cast_19, Celu
from onnx.reference.ops.aionnx_preview_training._op_list import Adam
from onnx.reference.ops.op_celu import _vcelu1
from onnx.reference.ops.op_col2im import (
from onnx.reference.ops.op_conv import Conv, _conv_implementation
from onnx.reference.ops_optimized import Conv as ConvOptimized
from onnx.reference.ops_optimized.op_conv_optimized import _conv_implementation_im2col
def test_onnxrt_reduce_mean(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    node1 = make_node('ReduceMean', ['X'], ['Y'])
    graph = make_graph([node1], 'g', [X], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 17)])
    check_model(onnx_model)
    sess = ReferenceEvaluator(onnx_model)
    cls = sess.rt_nodes_[0]
    self.assertEqual(cls.__class__.__name__, 'ReduceMean_1')
    got = sess.run(None, {'X': np.ones((2, 4), dtype=np.float32)})[0]
    self.assertEqual(got.shape, (1, 1))
    self.assertEqual(got[0, 0], 1)
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 18)])
    check_model(onnx_model)
    sess = ReferenceEvaluator(onnx_model)
    cls = sess.rt_nodes_[0]
    self.assertEqual(cls.__class__.__name__, 'ReduceMean_18')
    got = sess.run(None, {'X': np.ones((2, 4), dtype=np.float32)})[0]
    self.assertEqual(got.shape, (1, 1))
    self.assertEqual(got[0, 0], 1)