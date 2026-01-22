from __future__ import annotations
from typing import Any
from collections.abc import Iterable
from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify
from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.iterables import (has_dups, sift, iterable,
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name
import mpmath
from mpmath.libmp.libmpf import prec_to_dps
import inspect
from collections import Counter
from .symbol import Dummy, Symbol
def arity(cls):
    """Return the arity of the function if it is known, else None.

    Explanation
    ===========

    When default values are specified for some arguments, they are
    optional and the arity is reported as a tuple of possible values.

    Examples
    ========

    >>> from sympy import arity, log
    >>> arity(lambda x: x)
    1
    >>> arity(log)
    (1, 2)
    >>> arity(lambda *x: sum(x)) is None
    True
    """
    eval_ = getattr(cls, 'eval', cls)
    parameters = inspect.signature(eval_).parameters.items()
    if [p for _, p in parameters if p.kind == p.VAR_POSITIONAL]:
        return
    p_or_k = [p for _, p in parameters if p.kind == p.POSITIONAL_OR_KEYWORD]
    no, yes = map(len, sift(p_or_k, lambda p: p.default == p.empty, binary=True))
    return no if not yes else tuple(range(no, no + yes + 1))