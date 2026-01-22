import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def _make_deprecate(meth):
    new_name = meth.__name__
    old_name = new_name[:-1]

    def deprecated_init(*args, **kwargs):
        warnings.warn(f'nn.init.{old_name} is now deprecated in favor of nn.init.{new_name}.', stacklevel=2)
        return meth(*args, **kwargs)
    deprecated_init.__doc__ = f'\n    {old_name}(...)\n\n    .. warning::\n        This method is now deprecated in favor of :func:`torch.nn.init.{new_name}`.\n\n    See :func:`~torch.nn.init.{new_name}` for details.'
    deprecated_init.__name__ = old_name
    return deprecated_init