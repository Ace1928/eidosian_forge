from __future__ import annotations
from typing import Any, Callable
from inspect import getmro
import string
from sympy.core.random import choice
from .parameters import global_parameters
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
from .basic import Basic
def _convert_numpy_types(a, **sympify_args):
    """
    Converts a numpy datatype input to an appropriate SymPy type.
    """
    import numpy as np
    if not isinstance(a, np.floating):
        if np.iscomplex(a):
            return _sympy_converter[complex](a.item())
        else:
            return sympify(a.item(), **sympify_args)
    else:
        try:
            from .numbers import Float
            prec = np.finfo(a).nmant + 1
            a = str(list(np.reshape(np.asarray(a), (1, np.size(a)))[0]))[1:-1]
            return Float(a, precision=prec)
        except NotImplementedError:
            raise SympifyError('Translation for numpy float : %s is not implemented' % a)