import collections
from sympy.assumptions.ask import Q
from sympy.core.basic import (Basic, Atom, as_Basic,
from sympy.core.containers import Tuple
from sympy.core.function import Function, Lambda
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.concrete.summations import Sum
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.functions.elementary.exponential import exp
from sympy.testing.pytest import raises, warns_deprecated_sympy
class MyMeta(ManagedProperties):

    def __init__(cls, *args, **kwargs):
        myclasses.append('executed')
        super().__init__(*args, **kwargs)