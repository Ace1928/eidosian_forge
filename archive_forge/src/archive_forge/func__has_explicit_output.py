import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys
from types import SimpleNamespace
import numpy as np
import operator
from numba.core import types, targetconfig, ir, rewrites, compiler
from numba.core.typing import npydecl
from numba.np.ufunc.dufunc import DUFunc
def _has_explicit_output(self, expr, func):
    """
        Return whether the *expr* call to *func* (a ufunc) features an
        explicit output argument.
        """
    nargs = len(expr.args) + len(expr.kws)
    if expr.vararg is not None:
        return True
    return nargs > func.nin