import functools
import types
import warnings
import numpy as np
from .._utils import set_module
from . import multiarray as mu
from . import overrides
from . import umath as um
from . import numerictypes as nt
from .multiarray import asarray, array, asanyarray, concatenate
from . import _methods
def _mean_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None, *, where=None):
    return (a, where, out)