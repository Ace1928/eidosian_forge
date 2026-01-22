import math
import warnings
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .lazy import LazyModuleMixin
from .module import Module
from .utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes
from ..common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
def _get_num_spatial_dims(self) -> int:
    return 3