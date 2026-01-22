import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_adaptive_avg_pool2d(self, node):
    assert node.inputsSize() == 2
    assert node.outputsSize() == 1
    image_id, image_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
    assert len(image_oper.shape) == 4
    size_ctype, size_arg = self.get_constant_value(node.inputsAt(1))
    assert size_ctype.kind() == 'ListType'
    assert size_ctype.getElementType().kind() == 'IntType'
    if size_arg != [1, 1]:
        raise Exception('NNAPI only supports adaptive_avg_pool2d with output size (1, 1).')
    out_shape = image_oper.shape[0:2] + tuple(size_arg)
    use_nchw = image_oper.use_nchw()
    inputs = [None] * 11
    inputs[0] = image_id
    inputs[1] = self.add_immediate_int_scalar(0)
    inputs[2] = self.add_immediate_int_scalar(0)
    inputs[3] = self.add_immediate_int_scalar(0)
    inputs[4] = self.add_immediate_int_scalar(0)
    inputs[5] = self.add_immediate_int_scalar(1)
    inputs[6] = self.add_immediate_int_scalar(1)
    inputs[7] = self.add_immediate_int_scalar(image_oper.shape[3])
    inputs[8] = self.add_immediate_int_scalar(image_oper.shape[2])
    inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
    inputs[10] = self.add_immediate_bool_scalar(use_nchw)
    outputs = [None] * 1
    outputs[0] = self.add_tensor_operand(node.outputsAt(0), image_oper._replace(shape=out_shape))
    self.add_operation(NNAPI_OperationCode.AVERAGE_POOL_2D, inputs, outputs)