import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.ao.nn.intrinsic import _FusedModule
from typing import Tuple, TypeVar, Union
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
 This works for both single qat conv, and the qat conv - relu modules
        to convert the qat module to a floating point module
        