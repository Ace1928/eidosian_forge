import warnings
import torch
from torch.overrides import get_default_nowrap_functions
def _set_data_mask(self, data, mask):
    self._masked_data = data
    self._masked_mask = mask
    self._validate_members()