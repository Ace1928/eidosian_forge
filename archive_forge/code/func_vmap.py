import torch
from functools import partial
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
@staticmethod
def vmap(info, in_dims, x, idx):
    x_bdim, _ = in_dims
    x = x.movedim(x_bdim, 1)
    return (ForwardHasDefaultArgs.apply(x, idx), 0)