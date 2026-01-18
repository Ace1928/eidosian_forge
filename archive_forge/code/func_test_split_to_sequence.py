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
def test_split_to_sequence(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, None)
    Y = make_tensor_value_info('Y', TensorProto.INT64, None)
    Z = make_tensor_value_info('Z', TensorProto.UNDEFINED, None)
    nodes = [make_node('SplitToSequence', ['X', 'Y'], ['Z'], axis=2)]
    model = make_model(make_graph(nodes, 'g', [X, Y], [Z]))
    ref = ReferenceEvaluator(model)
    data = np.arange(18).reshape((1, 3, 6)).astype(np.float32)
    indices = np.array(2, dtype=np.int64)
    got = ref.run(None, {'X': data, 'Y': indices})
    expected = [[np.array([[[0.0, 1.0], [6.0, 7.0], [12.0, 13.0]]], dtype=np.float32), np.array([[[2.0, 3.0], [8.0, 9.0], [14.0, 15.0]]], dtype=np.float32), np.array([[[4.0, 5.0], [10.0, 11.0], [16.0, 17.0]]], dtype=np.float32)]]
    self.assertEqual(len(expected[0]), len(got[0]))
    for a, b in zip(expected[0], got[0]):
        assert_allclose(a, b)