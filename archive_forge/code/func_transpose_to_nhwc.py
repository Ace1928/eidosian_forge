import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def transpose_to_nhwc(self, in_id, oper):
    if oper.shape[2:] != (1, 1):
        raise Exception('Automatic transpose only supported for H,W == 1,1')
    out_oper = oper._replace(dim_order=DimOrder.CHANNELS_LAST)
    inputs = [None] * 2
    inputs[0] = in_id
    inputs[1] = self.add_immediate_int_vector([0, 2, 3, 1])
    outputs = [None] * 1
    outputs[0] = self.add_anonymous_tensor_operand(out_oper)
    self.add_operation(NNAPI_OperationCode.TRANSPOSE, inputs, outputs)
    return (outputs[0], out_oper)