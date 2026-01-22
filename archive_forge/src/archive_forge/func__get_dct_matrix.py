import math
from typing import Tuple
import torch
import torchaudio
from torch import Tensor
def _get_dct_matrix(num_ceps: int, num_mel_bins: int) -> Tensor:
    dct_matrix = torchaudio.functional.create_dct(num_mel_bins, num_mel_bins, 'ortho')
    dct_matrix[:, 0] = math.sqrt(1 / float(num_mel_bins))
    dct_matrix = dct_matrix[:, :num_ceps]
    return dct_matrix