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
@skip_if_no_onnxruntime
def test_qlinearconv(self):
    x = make_tensor_value_info('x', TensorProto.UINT8, [None, None, None, None])
    w = make_tensor_value_info('w', TensorProto.UINT8, [None, None, None, None])
    y = make_tensor_value_info('y', TensorProto.UINT8, [None, None, None, None])
    x_scale = make_tensor_value_info('x_scale', TensorProto.FLOAT, [None])
    w_scale = make_tensor_value_info('w_scale', TensorProto.FLOAT, [None])
    y_scale = make_tensor_value_info('y_scale', TensorProto.FLOAT, [None])
    x_zero_point = make_tensor_value_info('x_zero_point', TensorProto.UINT8, [None])
    w_zero_point = make_tensor_value_info('w_zero_point', TensorProto.UINT8, [None])
    y_zero_point = make_tensor_value_info('y_zero_point', TensorProto.UINT8, [None])
    node = make_node('QLinearConv', ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'], ['y'])
    graph = make_graph([node], 'g', [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point], [y])
    onnx_model = make_model_gen_version(graph, opset_imports=[make_opsetid('', 16)])
    sess1 = run_ort_inference(onnx_model)
    if sess1 is None:
        return
    sess2 = ReferenceEvaluator(onnx_model)
    sH, sW = (3, 3)
    for i in range(sH):
        for j in range(sW):
            x = np.zeros((1, 1, sH, sW), dtype=np.uint8)
            x[0, 0, i, j] = 1.0
            with self.subTest(w='1x1', i=i, j=j):
                w = np.zeros((1, 1, 1, 1), dtype=np.uint8)
                w[0, 0, :, :] = 1
                feeds = {'x': x, 'x_scale': np.array([1], dtype=np.float32), 'x_zero_point': np.array([0], dtype=np.uint8), 'w': w, 'w_scale': np.array([1], dtype=np.float32), 'w_zero_point': np.array([0], dtype=np.uint8), 'y_scale': np.array([1], dtype=np.float32), 'y_zero_point': np.array([0], np.uint8)}
                expected = sess1.run(None, feeds)[0]
                got = sess2.run(None, feeds)[0]
                try:
                    assert_allclose(expected, got)
                except AssertionError as e:
                    raise e
            with self.subTest(w='3x3', i=i, j=j):
                w = np.zeros((1, 1, 3, 3), dtype=np.uint8)
                w[0, 0, :, :] = np.minimum(2 ** np.arange(9).reshape((3, -1)), 128)
                feeds = {'x': x, 'x_scale': np.array([1], dtype=np.float32), 'x_zero_point': np.array([0], dtype=np.uint8), 'w': w, 'w_scale': np.array([1], dtype=np.float32), 'w_zero_point': np.array([0], dtype=np.uint8), 'y_scale': np.array([1], dtype=np.float32), 'y_zero_point': np.array([0], np.uint8)}
                expected = sess1.run(None, feeds)[0]
                got = sess2.run(None, feeds)[0]
                assert_allclose(expected, got)
            with self.subTest(w='1x1', i=i, j=j):
                w = np.zeros((1, 1, 1, 1), dtype=np.uint8)
                w[0, 0, :, :] = 0
                feeds = {'x': x, 'x_scale': np.array([0.00369204697], dtype=np.float32), 'x_zero_point': np.array([132], dtype=np.uint8), 'w': w, 'w_scale': np.array([100.00172794575], dtype=np.float32), 'w_zero_point': np.array([255], dtype=np.uint8), 'y_scale': np.array([0.00162681262], dtype=np.float32), 'y_zero_point': np.array([132], np.uint8)}
                expected = sess1.run(None, feeds)[0]
                got = sess2.run(None, feeds)[0]
                assert_allclose(expected, got)
    x = np.array([[255, 174, 162, 25, 203, 168, 58], [15, 59, 237, 95, 129, 0, 64], [56, 242, 153, 221, 168, 12, 166], [232, 178, 186, 195, 237, 162, 237], [188, 39, 124, 77, 80, 102, 43], [127, 230, 21, 83, 41, 40, 134], [255, 154, 92, 141, 42, 148, 247]], dtype=np.uint8).reshape((1, 1, 7, 7))
    x_scale = np.array([0.00369204697], dtype=np.float32)
    x_zero_point = np.array([132], dtype=np.uint8)
    w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))
    w_scale = np.array([0.00172794575], dtype=np.float32)
    w_zero_point = np.array([255], dtype=np.uint8)
    y_scale = np.array([0.00162681262], dtype=np.float32)
    y_zero_point = np.array([123], dtype=np.uint8)
    feeds = {'x': x, 'x_scale': x_scale, 'x_zero_point': x_zero_point, 'w': w, 'w_scale': w_scale, 'w_zero_point': w_zero_point, 'y_scale': y_scale, 'y_zero_point': y_zero_point}
    expected = sess1.run(None, feeds)[0]
    got = sess2.run(None, feeds)[0]
    assert_allclose(expected, got)