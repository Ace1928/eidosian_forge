from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
def _flatten_params(self, flat_params: List[nn.Parameter]) -> None:
    """Flatten the managed parameters and replaced the original
        attributes with views to the flat params.
        """
    assert not self.is_flattened
    self.is_flattened = True
    assert len(self.flat_param_names) == len(flat_params), f'{len(self.flat_param_names)} vs. {len(flat_params)}'
    for n, flat_param in zip(self.flat_param_names, flat_params):
        self.register_parameter(n, flat_param)
    self.flat_params = flat_params
    for _, m, n in self._param_infos:
        delattr(m, n)
    for _, _, m, n, _, _ in self._shared_param_infos:
        delattr(m, n)
    self._unflatten_params_as_views()