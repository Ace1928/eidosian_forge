import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def _handle_conv_pool_flexible_input(self, out_id, jit_image, args, transpose):
    image_id, image_oper = self.get_tensor_operand_by_jitval(jit_image)
    batch, in_ch, in_h, in_w = image_oper.shape
    if batch == 0:
        self.forward_operand_shape(out_id, 0, image_id, 0)
    if in_ch == 0:
        raise Exception("Input channels can't be flexible")
    if transpose:
        if in_h == 0:
            self.compute_operand_shape(out_id, 2, f'({flex_name(image_id, 2)} - 1) * {args.stride_h} + {args.kernel_h} - {args.pad_t} - {args.pad_b}')
        if in_w == 0:
            self.compute_operand_shape(out_id, 3, f'({flex_name(image_id, 3)} - 1) * {args.stride_w} + {args.kernel_w} - {args.pad_l} - {args.pad_r}')
    else:
        if in_h == 0:
            self.compute_operand_shape(out_id, 2, f'({flex_name(image_id, 2)} - {args.kernel_h} + {args.pad_t} + {args.pad_b}) // {args.stride_h} + 1')
        if in_w == 0:
            self.compute_operand_shape(out_id, 3, f'({flex_name(image_id, 3)} - {args.kernel_w} + {args.pad_l} + {args.pad_r}) // {args.stride_w} + 1')