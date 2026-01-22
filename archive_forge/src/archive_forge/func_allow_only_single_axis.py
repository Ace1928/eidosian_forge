import operator
import torch
from . import _dtypes_impl
def allow_only_single_axis(axis):
    if axis is None:
        return axis
    if len(axis) != 1:
        raise NotImplementedError('does not handle tuple axis')
    return axis[0]