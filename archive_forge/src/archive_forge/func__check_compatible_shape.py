import numpy
from .. import registry
from ..compat import cublas, cupy, cupyx
from ..types import DeviceTypes
from ..util import (
from . import _custom_kernels
from .numpy_ops import NumpyOps
from .ops import Ops
def _check_compatible_shape(u, v):
    if u.shape != v.shape:
        msg = f'arrays have incompatible shapes: {u.shape} and {v.shape}'
        raise ValueError(msg)