import pickle
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar, Replacer
def _get_first_match(expr, pattern):
    from matchpy import ManyToOneMatcher, Pattern
    matcher = ManyToOneMatcher()
    matcher.add(Pattern(pattern))
    return next(iter(matcher.match(expr)))