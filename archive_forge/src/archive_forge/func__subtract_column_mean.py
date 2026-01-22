import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def _subtract_column_mean(tensor: Tensor, subtract_mean: bool) -> Tensor:
    if subtract_mean:
        col_means = torch.mean(tensor, dim=0).unsqueeze(0)
        tensor = tensor - col_means
    return tensor