import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_hardtanh(self, node):
    assert node.inputsSize() == 3
    assert node.outputsSize() == 1
    in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
    _, min_val = self.get_constant_value(node.inputsAt(1), 'FloatType')
    _, max_val = self.get_constant_value(node.inputsAt(2), 'FloatType')
    op_map = {(-1, 1): NNAPI_OperationCode.RELU1, (0, 6): NNAPI_OperationCode.RELU6}
    opcode = op_map.get((min_val, max_val))
    if opcode is None:
        raise Exception('NNAPI only supports hardtanh with args (-1, 1) or (0, 6).')
    inputs = [None] * 1
    inputs[0] = in_id
    outputs = [None] * 1
    outputs[0] = self.add_tensor_operand(node.outputsAt(0), in_oper)
    self.add_operation(opcode, inputs, outputs)