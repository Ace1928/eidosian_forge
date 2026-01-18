import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
@parameterized.parameterized.expand([('per_tensor', (16, 3), (1,), None, None, None, TensorProto.INT8, True), ('per_axis_none_block_shape', (16, 3), (16,), 1, None, None, TensorProto.INT8, True), ('per_axis_zero_block_shape', (16, 3), (16,), 1, 0, None, TensorProto.INT8, True), ('per_tensor_positive_block_shape', (16, 3), (1,), 1, 2, None, TensorProto.INT8, False), ('per_axis_positive_block_shape', (16, 3), (16,), 1, 2, None, TensorProto.INT8, False), ('blocked_2d', (16, 3), (4, 3), 0, 4, None, TensorProto.INT8, False), ('blocked_3d', (4, 3, 32), (4, 3, 8), 2, 4, None, TensorProto.INT8, False), ('per_axis_output_dtype', (16, 3), (16,), 1, None, TensorProto.FLOAT8E4M3FN, None, False), ('per_axis_unsupported_type', (16, 3), (16,), 1, None, None, TensorProto.UINT16, False)])
def test_quantize_21_20(self, _: str, x_shape: Tuple[int, ...], scale_shape: Tuple[int, ...], axis: int, block_size: int, output_dtype: Optional[int], zero_point_dtype: Optional[int], compatible: bool) -> None:

    def test(input_shape, scale_shape, axis, block_size, output_dtype, zero_point_dtype) -> None:
        nodes = [helper.make_node('QuantizeLinear', ['X', 'S'], ['Y'], axis=axis, block_size=block_size, output_dtype=output_dtype)]
        inputs = [helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape), helper.make_tensor_value_info('S', TensorProto.FLOAT, scale_shape)]
        if zero_point_dtype:
            inputs.append(helper.make_tensor_value_info('ZP', zero_point_dtype, scale_shape))
            nodes[0].input.append('ZP')
        output_type_ = output_dtype or zero_point_dtype
        graph = helper.make_graph(nodes, 'test', inputs, [helper.make_tensor_value_info('Y', output_type_, input_shape)])
        _ = self._converted(graph, helper.make_operatorsetid('', 21), 20)
    context_manager = contextlib.nullcontext() if compatible else self.assertRaises(RuntimeError)
    with context_manager:
        test(x_shape, scale_shape, axis, block_size, output_dtype, zero_point_dtype)