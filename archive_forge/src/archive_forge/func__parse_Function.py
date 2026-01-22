from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable
import sympy
from sympy import Mul, Add, Pow, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning
def _parse_Function(*args):
    if len(args) == 1:
        arg = args[0]
        Slot = Function('Slot')
        slots = arg.atoms(Slot)
        numbers = [a.args[0] for a in slots]
        number_of_arguments = max(numbers)
        if isinstance(number_of_arguments, Integer):
            variables = symbols(f'dummy0:{number_of_arguments}', cls=Dummy)
            return Lambda(variables, arg.xreplace({Slot(i + 1): v for i, v in enumerate(variables)}))
        return Lambda((), arg)
    elif len(args) == 2:
        variables = args[0]
        body = args[1]
        return Lambda(variables, body)
    else:
        raise SyntaxError('Function node expects 1 or 2 arguments')