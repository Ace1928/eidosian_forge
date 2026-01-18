import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def vectorwise_dequant(xq, max1, quant_type='vector'):
    if quant_type == 'vector':
        x = (xq / C * max1).to(torch.float32)
        return x
    else:
        return None