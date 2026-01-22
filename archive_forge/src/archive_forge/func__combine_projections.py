import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
def _combine_projections(self, a: torch.Tensor, b: torch.Tensor, _inplace_chunk_size: Optional[int]=None) -> torch.Tensor:
    if self._outgoing:
        a = permute_final_dims(a, (2, 0, 1))
        b = permute_final_dims(b, (2, 1, 0))
    else:
        a = permute_final_dims(a, (2, 1, 0))
        b = permute_final_dims(b, (2, 0, 1))
    if _inplace_chunk_size is not None:
        for i in range(0, a.shape[-3], _inplace_chunk_size):
            a_chunk = a[..., i:i + _inplace_chunk_size, :, :]
            b_chunk = b[..., i:i + _inplace_chunk_size, :, :]
            a[..., i:i + _inplace_chunk_size, :, :] = torch.matmul(a_chunk, b_chunk)
        p = a
    else:
        p = torch.matmul(a, b)
    return permute_final_dims(p, (1, 2, 0))