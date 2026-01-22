import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
def _ispec(self, z, length=None):
    hl = self.hop_length
    z = F.pad(z, [0, 0, 0, 1])
    z = F.pad(z, [2, 2])
    pad = hl // 2 * 3
    le = hl * int(math.ceil(length / hl)) + 2 * pad
    x = _ispectro(z, hl, length=le)
    x = x[..., pad:pad + length]
    return x