import math
import random
from typing import List, Optional, Sequence, Tuple, Type
import torch
from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionOpBase
def _create_aligned_bias(*shape: int, **kwargs) -> torch.Tensor:
    align_to = 8
    return (torch.randn((*shape[:-1], align_to * ((shape[-1] + align_to - 1) // align_to)), **kwargs) * 3).narrow(-1, 0, shape[-1])