from typing import Optional, Iterable
import torch
from math import sqrt
from torch import Tensor
from torch._torch_docs import factory_common_args, parse_kwargs, merge_dicts
def _add_docstr(*args):
    """Adds docstrings to a given decorated function.

    Specially useful when then docstrings needs string interpolation, e.g., with
    str.format().
    REMARK: Do not use this function if the docstring doesn't need string
    interpolation, just write a conventional docstring.

    Args:
        args (str):
    """

    def decorator(o):
        o.__doc__ = ''.join(args)
        return o
    return decorator