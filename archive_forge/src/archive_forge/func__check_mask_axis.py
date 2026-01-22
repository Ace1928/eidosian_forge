import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def _check_mask_axis(mask, axis, keepdims=np._NoValue):
    """Check whether there are masked values along the given axis"""
    kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
    if mask is not nomask:
        return mask.all(axis=axis, **kwargs)
    return nomask