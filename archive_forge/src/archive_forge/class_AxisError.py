import operator
import torch
from . import _dtypes_impl
class AxisError(ValueError, IndexError):
    pass