import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
def _hsv2rgb(img: Tensor) -> Tensor:
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.to(dtype=torch.int32)
    p = torch.clamp(v * (1.0 - s), 0.0, 1.0)
    q = torch.clamp(v * (1.0 - s * f), 0.0, 1.0)
    t = torch.clamp(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)
    i = i % 6
    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)
    return torch.einsum('...ijk, ...xijk -> ...xjk', mask.to(dtype=img.dtype), a4)