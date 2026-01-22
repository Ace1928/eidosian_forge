from __future__ import annotations
from .assumptions import StdFactKB, _assume_defined
from .basic import Basic, Atom
from .cache import cacheit
from .containers import Tuple
from .expr import Expr, AtomicExpr
from .function import AppliedUndef, FunctionClass
from .kind import NumberKind, UndefinedKind
from .logic import fuzzy_bool
from .singleton import S
from .sorting import ordered
from .sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.misc import filldedent
import string
import re as _re
import random
from itertools import product
from typing import Any
@staticmethod
@cacheit
def __xnew_cached_(cls, name, **assumptions):
    return Symbol.__xnew__(cls, name, **assumptions)