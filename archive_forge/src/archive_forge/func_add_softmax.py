import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_softmax(self, node):
    assert node.inputsSize() == 3
    in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
    _, softmax_dim = self.get_constant_value(node.inputsAt(1), 'IntType')
    out_id = self.add_tensor_operand(node.outputsAt(0), in_oper)
    for dim, size in enumerate(in_oper.shape):
        if size == 0:
            self.forward_operand_shape(out_id, dim, in_id, dim)
    inputs = [None] * 3
    inputs[0] = in_id
    inputs[1] = self.add_immediate_float_scalar(1.0)
    inputs[2] = self.add_immediate_int_scalar(softmax_dim)
    outputs = [None] * 1
    outputs[0] = out_id
    self.add_operation(NNAPI_OperationCode.SOFTMAX, inputs, outputs)