from __future__ import annotations
from typing import Any, Callable
from inspect import getmro
import string
from sympy.core.random import choice
from .parameters import global_parameters
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable
from .basic import Basic
Use a hack to try keep autosimplification from distributing a
    a number into an Add; this modification does not
    prevent the 2-arg Mul from becoming an Add, however.

    Examples
    ========

    >>> from sympy.core.sympify import kernS
    >>> from sympy.abc import x, y

    The 2-arg Mul distributes a number (or minus sign) across the terms
    of an expression, but kernS will prevent that:

    >>> 2*(x + y), -(x + 1)
    (2*x + 2*y, -x - 1)
    >>> kernS('2*(x + y)')
    2*(x + y)
    >>> kernS('-(x + 1)')
    -(x + 1)

    If use of the hack fails, the un-hacked string will be passed to sympify...
    and you get what you get.

    XXX This hack should not be necessary once issue 4596 has been resolved.
    