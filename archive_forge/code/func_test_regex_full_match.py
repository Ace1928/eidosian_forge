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
@parameterized.parameterized.expand([(['www.google.com', 'www.facebook.com', 'www.bbc.co.uk'], 'www\\.[\\w.-]+\\.\\bcom\\b', [True, True, False], (3,)), ([['Onnx', 'tensorflow', 'Numpy'], ['Pytorch', 'Cython', 'numba']], '^[A-Z][a-z]*$', [[True, False, True], [True, True, False]], (2, 3)), (['account@gmail.com', 'account@hotmail.com', 'not email', 'account2@yahoo.com'], '(\\W|^)[\\w.\\-]{0,25}@(yahoo|gmail)\\.com(\\W|$)', [True, False, False, True], (4,))])
@unittest.skipIf(sys.platform == 'win32', 'google-re2 package is not built for win32')
def test_regex_full_match(self, x, pattern, expected, expected_shape):
    X = make_tensor_value_info('X', TensorProto.STRING, None)
    Y = make_tensor_value_info('Y', TensorProto.BOOL, None)
    node = make_node('RegexFullMatch', inputs=['X'], outputs=['Y'], pattern=pattern)
    model = make_model(make_graph([node], 'g', [X], [Y]))
    ref = ReferenceEvaluator(model)
    result, *_ = ref.run(None, {'X': np.array(x)})
    np.testing.assert_array_equal(result, expected)
    self.assertEqual(result.dtype.kind, 'b')
    self.assertEqual(result.shape, expected_shape)