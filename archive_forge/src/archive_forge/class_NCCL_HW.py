import math
from enum import IntEnum
from typing import TYPE_CHECKING
import torch
from . import ir
from .utils import get_dtype_size, sympy_product
from .virtualized import V
class NCCL_HW(IntEnum):
    NVLINK = 0
    PCI = 1
    NET = 2