from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping
from itertools import chain, zip_longest
from .assumptions import _prepare_class_assumptions
from .cache import cacheit
from .core import ordering_of_classes
from .sympify import _sympify, sympify, SympifyError, _external_converter
from .sorting import ordered
from .kind import Kind, UndefinedKind
from ._print_helpers import Printable
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, numbered_symbols
from sympy.utilities.misc import filldedent, func_name
from inspect import getmro
from .singleton import S
from .traversal import (preorder_traversal as _preorder_traversal,
def _sage_(self):
    """
        Convert *self* to a symbolic expression of SageMath.

        This version of the method is merely a placeholder.
        """
    old_method = self._sage_
    from sage.interfaces.sympy import sympy_init
    sympy_init()
    if old_method == self._sage_:
        raise NotImplementedError('conversion to SageMath is not implemented')
    else:
        return self._sage_()