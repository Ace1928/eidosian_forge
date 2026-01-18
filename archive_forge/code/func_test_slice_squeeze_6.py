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
def test_slice_squeeze_6(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.INT64, [None])
    nodes = [make_node('Slice', ['X'], ['T'], axes=[0], starts=[0], ends=[1]), make_node('Squeeze', ['T'], ['Y'], axes=[0])]
    graph = make_graph(nodes, 'g', [X], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 6)])
    feeds = {'X': np.array([[0]], dtype=np.int64)}
    ref1 = ReferenceEvaluator(onnx_model)
    got1 = ref1.run(None, feeds)
    expected = np.array([0], dtype=np.int64)
    self.assertEqual(expected.tolist(), got1[0].tolist())