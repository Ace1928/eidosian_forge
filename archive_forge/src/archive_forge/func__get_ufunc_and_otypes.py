import collections.abc
import functools
import re
import sys
import warnings
from .._utils import set_module
import numpy as np
import numpy.core.numeric as _nx
from numpy.core import transpose
from numpy.core.numeric import (
from numpy.core.umath import (
from numpy.core.fromnumeric import (
from numpy.core.numerictypes import typecodes
from numpy.core import overrides
from numpy.core.function_base import add_newdoc
from numpy.lib.twodim_base import diag
from numpy.core.multiarray import (
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
import builtins
from numpy.lib.histograms import histogram, histogramdd  # noqa: F401
def _get_ufunc_and_otypes(self, func, args):
    """Return (ufunc, otypes)."""
    if not args:
        raise ValueError('args can not be empty')
    if self.otypes is not None:
        otypes = self.otypes
        nin = len(args)
        nout = len(self.otypes)
        if func is not self.pyfunc or nin not in self._ufunc:
            ufunc = frompyfunc(func, nin, nout)
        else:
            ufunc = None
        if func is self.pyfunc:
            ufunc = self._ufunc.setdefault(nin, ufunc)
    else:
        args = [asarray(arg) for arg in args]
        if builtins.any((arg.size == 0 for arg in args)):
            raise ValueError('cannot call `vectorize` on size 0 inputs unless `otypes` is set')
        inputs = [arg.flat[0] for arg in args]
        outputs = func(*inputs)
        if self.cache:
            _cache = [outputs]

            def _func(*vargs):
                if _cache:
                    return _cache.pop()
                else:
                    return func(*vargs)
        else:
            _func = func
        if isinstance(outputs, tuple):
            nout = len(outputs)
        else:
            nout = 1
            outputs = (outputs,)
        otypes = ''.join([asarray(outputs[_k]).dtype.char for _k in range(nout)])
        ufunc = frompyfunc(_func, len(args), nout)
    return (ufunc, otypes)