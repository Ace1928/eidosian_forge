import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def torch_tensor_to_operand(self, tensor, dim_order):
    dtype = str(tensor.dtype).replace('torch.', '')
    scale = 0.0
    zero_point = 0
    if dtype == 'float32':
        op_type = NNAPI_OperandCode.TENSOR_FLOAT32
    elif dtype == 'int32':
        op_type = NNAPI_OperandCode.TENSOR_INT32
    elif dtype == 'quint8':
        op_type = NNAPI_OperandCode.TENSOR_QUANT8_ASYMM
        scale = tensor.q_scale()
        zero_point = tensor.q_zero_point()
    elif dtype == 'qint32':
        op_type = NNAPI_OperandCode.TENSOR_INT32
        scale = tensor.q_scale()
        zero_point = tensor.q_zero_point()
        assert zero_point == 0
    elif dtype == 'int16':
        if self.use_int16_for_qint16:
            nnapi_dtype = getattr(tensor, 'nnapi_dtype', None)
            op_codes = (NNAPI_OperandCode.TENSOR_QUANT16_SYMM, NNAPI_OperandCode.TENSOR_QUANT16_ASYMM)
            if nnapi_dtype in op_codes:
                op_type = nnapi_dtype
                scale = tensor.nnapi_scale
                zero_point = tensor.nnapi_zero_point
            else:
                raise Exception(f'`nnapi_type` needs to be one of {op_codes} for `int16`')
        else:
            raise Exception("`int16` isn't supported. If you're trying to represent NNAPI qint16 with Pytorch int16, set `use_int16_for_qint16 = True`")
    else:
        raise Exception(f"Can't handle input with dtype '{tensor.dtype}'")
    return Operand(shape=tuple(tensor.shape), op_type=op_type, dim_order=dim_order, scale=scale, zero_point=zero_point)