from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
def _print_DenseMatrix(self, X, **kwargs):
    if not hasattr(tt, 'stacklists'):
        raise NotImplementedError('Matrix translation not yet supported in this version of Theano')
    return tt.stacklists([[self._print(arg, **kwargs) for arg in L] for L in X.tolist()])