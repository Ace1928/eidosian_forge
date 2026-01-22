import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_addmm_or_linear(self, node, transpose_weight, jit_input, jit_weight, jit_bias):
    input_id, input_oper = self.get_tensor_operand_by_jitval(jit_input)
    bias_id, bias_oper = self.get_tensor_operand_for_weight(jit_bias)
    assert len(input_oper.shape) == 2
    assert len(bias_oper.shape) == 1
    _, weight_tensor = self.get_constant_value(jit_weight, 'TensorType')
    assert len(weight_tensor.shape) == 2
    if transpose_weight:
        nnapi_weight_tensor = weight_tensor.t().contiguous()
    else:
        nnapi_weight_tensor = weight_tensor.contiguous()
    weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
    weight_oper = self.operands[weight_id]
    out_shape = (input_oper.shape[0], weight_oper.shape[0])
    out_id = self.add_tensor_operand(node.outputsAt(0), input_oper._replace(shape=out_shape))
    if input_oper.shape[0] == 0:
        self.forward_operand_shape(out_id, 0, input_id, 0)
    inputs = [None] * 4
    inputs[0] = input_id
    inputs[1] = weight_id
    inputs[2] = bias_id
    inputs[3] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
    outputs = [None] * 1
    outputs[0] = out_id
    self.add_operation(NNAPI_OperationCode.FULLY_CONNECTED, inputs, outputs)