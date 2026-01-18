from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def undiff(g):
    if g.shape[axis] > 0:
        return anp.concatenate((-g[tuple(sl1)], -anp.diff(g, axis=axis), g[tuple(sl2)]), axis=axis)
    shape = list(ans_shape)
    shape[axis] = 1
    return anp.zeros(shape)