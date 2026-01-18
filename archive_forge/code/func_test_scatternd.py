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
def test_scatternd(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Ind = make_tensor_value_info('I', TensorProto.INT64, [None, None])
    U = make_tensor_value_info('U', TensorProto.FLOAT, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    node = make_node('ScatterND', ['X', 'I', 'U'], ['Y'])
    graph = make_graph([node], 'g', [X, Ind, U], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
    feeds = {'X': np.array([[1.0, 2.0]], dtype=np.float32), 'I': np.array([[0, 0]]), 'U': np.array([3.0], dtype=np.float32)}
    ref1 = ReferenceEvaluator(onnx_model)
    got1 = ref1.run(None, feeds)
    expected = np.array([[3.0, 2.0]], dtype=np.float32)
    assert_allclose(expected, got1[0])