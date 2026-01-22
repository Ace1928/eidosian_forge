import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_reshape(self, node):
    assert node.inputsSize() == 2
    assert node.outputsSize() == 1
    in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
    shape_ctype, shape = self.get_constant_value(node.inputsAt(1))
    assert shape_ctype.kind() == 'ListType'
    assert shape_ctype.getElementType().kind() == 'IntType'
    is_trivial_reshape = len(shape) == 2 and shape[1] == -1
    if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS and (not is_trivial_reshape):
        raise Exception('Currently, reshape is only supported on NHWC tensors if the target size is [X, -1].')
    out_shape = torch.zeros(1).expand(in_oper.shape).reshape(shape).shape
    out_oper = in_oper._replace(shape=out_shape, dim_order=DimOrder.PRESUMED_CONTIGUOUS)
    inputs = [None] * 2
    inputs[0] = in_id
    inputs[1] = self.add_immediate_int_vector(shape)
    outputs = [None] * 1
    outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)
    self.add_operation(NNAPI_OperationCode.RESHAPE, inputs, outputs)