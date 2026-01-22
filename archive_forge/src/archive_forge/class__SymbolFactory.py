from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
class _SymbolFactory:

    def __init__(self, label):
        self._counterVar = 0
        self._label = label

    def _set_counter(self, value):
        """
        Sets counter to value.
        """
        self._counterVar = value

    @property
    def _counter(self):
        """
        What counter is currently at.
        """
        return self._counterVar

    def _next(self):
        """
        Generates the next symbols and increments counter by 1.
        """
        s = Symbol('%s%i' % (self._label, self._counterVar))
        self._counterVar += 1
        return s