from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def unbroadcast_einsum(x, target_meta, subscript):
    if Ellipsis not in subscript:
        return x
    elif subscript[0] == Ellipsis:
        return unbroadcast(x, target_meta, 0)
    elif subscript[-1] == Ellipsis:
        return unbroadcast(x, target_meta, -1)
    else:
        return unbroadcast(x, target_meta, subscript.index(Ellipsis))