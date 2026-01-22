import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_quantize(self, node):
    assert node.inputsSize() == 4
    assert node.outputsSize() == 1
    in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
    if in_oper.dim_order != DimOrder.CHANNELS_LAST:
        raise Exception('Most hardware backends prefer NHWC quantized tensors.  Try setting `t.nnapi_nhwc = True` on your tensor inputs.  ')
    _, scale = self.get_constant_value(node.inputsAt(1), 'FloatType')
    _, zero_point = self.get_constant_value(node.inputsAt(2), 'IntType')
    _, scalar_type = self.get_constant_value(node.inputsAt(3), 'IntType')
    if scalar_type != TorchScalarTypes.QUINT8.value:
        raise Exception('PyTorch NNAPI export only supports quantized tensors with the quint8 dtype.')
    op_type = NNAPI_OperandCode.TENSOR_QUANT8_ASYMM
    out_oper = in_oper._replace(op_type=op_type, scale=scale, zero_point=zero_point)
    inputs = [None] * 1
    inputs[0] = in_id
    outputs = [None] * 1
    outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)
    self.add_operation(NNAPI_OperationCode.QUANTIZE, inputs, outputs)