import collections.abc
import math
import warnings
from typing import cast, List, Optional, Tuple, Union
import torch
def _uniform_random_(t: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if high - low >= torch.finfo(t.dtype).max:
        return t.uniform_(low / 2, high / 2).mul_(2)
    else:
        return t.uniform_(low, high)