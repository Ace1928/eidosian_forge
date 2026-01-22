from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
def _print_AppliedUndef(self, s, **kwargs):
    name = str(type(s)) + '_' + str(s.args[0])
    dtype = kwargs.get('dtypes', {}).get(s)
    bc = kwargs.get('broadcastables', {}).get(s)
    return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)