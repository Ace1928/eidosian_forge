from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
Validate the arguments for speech_reverberation_modulation_energy_ratio.

    Args:
        fs: the sampling rate
        n_cochlear_filters: Number of filters in the acoustic filterbank
        low_freq: determines the frequency cutoff for the corresponding gammatone filterbank.
        min_cf: Center frequency in Hz of the first modulation filter.
        max_cf: Center frequency in Hz of the last modulation filter. If None is given,
        norm: Use modulation spectrum energy normalization
        fast: Use the faster version based on the gammatonegram.

    