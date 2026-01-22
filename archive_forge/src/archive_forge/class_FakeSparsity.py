from typing import Any, Dict, Optional, Type
from torch.nn.utils.parametrize import type_before_parametrizations, is_parametrized
from itertools import chain
from torch import nn
class FakeSparsity(nn.Module):
    """Parametrization for the weights. Should be attached to the 'weight' or
    any other parameter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer('mask', mask)

    def forward(self, x):
        assert self.mask.shape == x.shape
        return self.mask * x

    def state_dict(self, *args, **kwargs):
        return {}