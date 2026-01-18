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
def test_quantize_linear_int16(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None])
    Y = make_tensor_value_info('Y', TensorProto.INT16, [None])
    model = make_model(make_graph([make_node('QuantizeLinear', ['X', 'scale', 'zero'], ['Y'])], 'g', [X], [Y], [make_tensor('scale', TensorProto.FLOAT, [1], [2.0]), make_tensor('zero', TensorProto.INT16, [1], [256])]))
    ref = ReferenceEvaluator(model)
    data = np.array([0.0, -514.0, 3.0, -3.0, 2.9, -2.9, 3.1, -3.1, 65022.0, -66046.0, 65023.0, -66047.0, 65024.0, -66048.0, 70000.0, -70000.0], dtype=np.float32)
    expected = np.array([256, -1, 258, 254, 257, 255, 258, 254, 32767, -32767, 32767, -32768, 32767, -32768, 32767, -32768], dtype=np.int16)
    got = ref.run(None, {'X': data})
    assert_allclose(expected, got[0])