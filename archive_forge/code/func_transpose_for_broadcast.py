import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def transpose_for_broadcast(self, in0_id, in0_oper, in1_id, in1_oper):
    if in0_oper.dim_order == in1_oper.dim_order:
        return (in0_id, in0_oper, in1_id, in1_oper)
    orders = (in0_oper.dim_order, in1_oper.dim_order)
    if orders == (DimOrder.PRESUMED_CONTIGUOUS, DimOrder.CHANNELS_LAST):
        return self.transpose_to_nhwc(in0_id, in0_oper) + (in1_id, in1_oper)
    if orders == (DimOrder.CHANNELS_LAST, DimOrder.PRESUMED_CONTIGUOUS):
        return (in0_id, in0_oper) + self.transpose_to_nhwc(in1_id, in1_oper)
    raise Exception(f'Automatic transpose not supported for dim_orders: {in0_oper.dim_order!r}, {in1_oper.dim_order!r}')