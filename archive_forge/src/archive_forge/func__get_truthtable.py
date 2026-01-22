from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
def _get_truthtable(variables, expr, const):
    """ Return a list of all combinations leading to a True result for ``expr``.
    """
    _variables = variables.copy()

    def _get_tt(inputs):
        if _variables:
            v = _variables.pop()
            tab = [[i[0].xreplace({v: false}), [0] + i[1]] for i in inputs if i[0] is not false]
            tab.extend([[i[0].xreplace({v: true}), [1] + i[1]] for i in inputs if i[0] is not false])
            return _get_tt(tab)
        return inputs
    res = [const + k[1] for k in _get_tt([[expr, []]]) if k[0]]
    if res == [[]]:
        return []
    else:
        return res