import operator
import torch
from . import _dtypes_impl
class UFuncTypeError(TypeError, RuntimeError):
    pass