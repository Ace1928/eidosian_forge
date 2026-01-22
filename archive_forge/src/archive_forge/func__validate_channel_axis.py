from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
def _validate_channel_axis(channel_axis, ndim):
    if not isinstance(channel_axis, int):
        raise TypeError('channel_axis must be an integer')
    if channel_axis < -ndim or channel_axis >= ndim:
        raise AxisError('channel_axis exceeds array dimensions')