from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
def _unflatten_params(self, external_data: Optional[List[Optional[Tensor]]]=None) -> None:
    """Undo flattening and create separate parameters from the already flattened
        self.flat_param or a user supplied external data.
        """
    assert self.is_flattened or external_data is not None
    self.is_flattened = False
    ps = self.get_param_views(external_data)
    for (_, m, n), p in zip(self._param_infos, ps):
        if hasattr(m, n):
            delattr(m, n)
        m.register_parameter(n, nn.Parameter(p))
    for _, _, m, n, shared_m, shared_n in self._shared_param_infos:
        if hasattr(m, n):
            delattr(m, n)
        m.register_parameter(n, getattr(shared_m, shared_n))
    if hasattr(self._fpw_module, '_unflattened_param_views'):
        delattr(self._fpw_module, '_unflattened_param_views')
    for n in self.flat_param_names:
        delattr(self, n)
    self.flat_params = []