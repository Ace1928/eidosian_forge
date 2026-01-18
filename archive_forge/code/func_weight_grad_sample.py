import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def weight_grad_sample(weight):
    if batch_size < THRESHOLD and groups == 1:
        return conv_group_weight_grad_sample(ctx.input, grad_output, weight_shape, stride, padding, dilation, batch_size, func)
    else:
        return conv_unfold_weight_grad_sample(ctx.input, grad_output, weight_shape, kernel_size, stride, padding, dilation, groups, func)