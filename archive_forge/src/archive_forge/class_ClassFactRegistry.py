from collections import defaultdict
from sympy.assumptions.ask import Q
from sympy.core import (Add, Mul, Pow, Number, NumberSymbol, Symbol)
from sympy.core.numbers import ImaginaryUnit
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import (Equivalent, And, Or, Implies)
from sympy.matrices.expressions import MatMul
class ClassFactRegistry:
    """
    Register handlers against classes.

    Explanation
    ===========

    ``register`` method registers the handler function for a class. Here,
    handler function should return a single fact. ``multiregister`` method
    registers the handler function for multiple classes. Here, handler function
    should return a container of multiple facts.

    ``registry(expr)`` returns a set of facts for *expr*.

    Examples
    ========

    Here, we register the facts for ``Abs``.

    >>> from sympy import Abs, Equivalent, Q
    >>> from sympy.assumptions.sathandlers import ClassFactRegistry
    >>> reg = ClassFactRegistry()
    >>> @reg.register(Abs)
    ... def f1(expr):
    ...     return Q.nonnegative(expr)
    >>> @reg.register(Abs)
    ... def f2(expr):
    ...     arg = expr.args[0]
    ...     return Equivalent(~Q.zero(arg), ~Q.zero(expr))

    Calling the registry with expression returns the defined facts for the
    expression.

    >>> from sympy.abc import x
    >>> reg(Abs(x))
    {Q.nonnegative(Abs(x)), Equivalent(~Q.zero(x), ~Q.zero(Abs(x)))}

    Multiple facts can be registered at once by ``multiregister`` method.

    >>> reg2 = ClassFactRegistry()
    >>> @reg2.multiregister(Abs)
    ... def _(expr):
    ...     arg = expr.args[0]
    ...     return [Q.even(arg) >> Q.even(expr), Q.odd(arg) >> Q.odd(expr)]
    >>> reg2(Abs(x))
    {Implies(Q.even(x), Q.even(Abs(x))), Implies(Q.odd(x), Q.odd(Abs(x)))}

    """

    def __init__(self):
        self.singlefacts = defaultdict(frozenset)
        self.multifacts = defaultdict(frozenset)

    def register(self, cls):

        def _(func):
            self.singlefacts[cls] |= {func}
            return func
        return _

    def multiregister(self, *classes):

        def _(func):
            for cls in classes:
                self.multifacts[cls] |= {func}
            return func
        return _

    def __getitem__(self, key):
        ret1 = self.singlefacts[key]
        for k in self.singlefacts:
            if issubclass(key, k):
                ret1 |= self.singlefacts[k]
        ret2 = self.multifacts[key]
        for k in self.multifacts:
            if issubclass(key, k):
                ret2 |= self.multifacts[k]
        return (ret1, ret2)

    def __call__(self, expr):
        ret = set()
        handlers1, handlers2 = self[type(expr)]
        for h in handlers1:
            ret.add(h(expr))
        for h in handlers2:
            ret.update(h(expr))
        return ret