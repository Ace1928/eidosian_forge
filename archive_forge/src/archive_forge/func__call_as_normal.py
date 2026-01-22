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
def _call_as_normal(self, *args, **kwargs):
    """
        Return arrays with the results of `pyfunc` broadcast (vectorized) over
        `args` and `kwargs` not in `excluded`.
        """
    excluded = self.excluded
    if not kwargs and (not excluded):
        func = self.pyfunc
        vargs = args
    else:
        nargs = len(args)
        names = [_n for _n in kwargs if _n not in excluded]
        inds = [_i for _i in range(nargs) if _i not in excluded]
        the_args = list(args)

        def func(*vargs):
            for _n, _i in enumerate(inds):
                the_args[_i] = vargs[_n]
            kwargs.update(zip(names, vargs[len(inds):]))
            return self.pyfunc(*the_args, **kwargs)
        vargs = [args[_i] for _i in inds]
        vargs.extend([kwargs[_n] for _n in names])
    return self._vectorize_call(func=func, args=vargs)