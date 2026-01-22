import math
import warnings
from typing import Any, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from peft.tuners.lycoris_utils import LycorisLayer, check_adapters_to_merge
def _project_batch(self, oft_r, eps=1e-05):
    eps = eps * 1 / torch.sqrt(torch.tensor(oft_r.shape[0]))
    I = torch.zeros((oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype).unsqueeze(0).expand_as(oft_r)
    diff = oft_r - I
    norm_diff = torch.norm(oft_r - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, oft_r, I + eps * (diff / norm_diff))
    return out