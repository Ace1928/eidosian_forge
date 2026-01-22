from itertools import permutations
from typing import Any, Callable, Tuple
import numpy as np
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def _gen_permutations(spk_num: int, device: torch.device) -> Tensor:
    key = str(spk_num) + str(device)
    if key not in _ps_dict:
        ps = torch.tensor(list(permutations(range(spk_num))), device=device)
        _ps_dict[key] = ps
    else:
        ps = _ps_dict[key]
    return ps