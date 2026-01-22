import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_avg_pool2d(self, node):
    assert node.inputsSize() == 7
    assert node.outputsSize() == 1
    image, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override = node.inputs()
    _, count_include_pad_value = self.get_constant_value(count_include_pad)
    _, divisor_override_value = self.get_constant_value(divisor_override)
    if not count_include_pad_value or divisor_override_value:
        raise Exception("NNAPI doesn't support count_include_pad=False or divisor_override")
    args = self.get_conv_pool_args_2d_from_jit(self.get_size_arg(kernel), stride, padding)
    image_id, image_oper = self.get_tensor_operand_by_jitval(image)
    assert len(image_oper.shape) == 4
    out_shape = get_conv_pool_shape(image_oper.shape, args, image_oper.shape[1], False)
    use_nchw = image_oper.use_nchw()
    inputs = [None] * 11
    inputs[0] = image_id
    inputs[1] = self.add_immediate_int_scalar(args.pad_l)
    inputs[2] = self.add_immediate_int_scalar(args.pad_r)
    inputs[3] = self.add_immediate_int_scalar(args.pad_t)
    inputs[4] = self.add_immediate_int_scalar(args.pad_b)
    inputs[5] = self.add_immediate_int_scalar(args.stride_w)
    inputs[6] = self.add_immediate_int_scalar(args.stride_h)
    inputs[7] = self.add_immediate_int_scalar(args.kernel_w)
    inputs[8] = self.add_immediate_int_scalar(args.kernel_h)
    inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
    inputs[10] = self.add_immediate_bool_scalar(use_nchw)
    outputs = [None] * 1
    out_id = self.add_tensor_operand(node.outputsAt(0), image_oper._replace(shape=out_shape))
    self._handle_conv_pool_flexible_input(out_id, image, args, False)
    outputs[0] = out_id
    self.add_operation(NNAPI_OperationCode.AVERAGE_POOL_2D, inputs, outputs)