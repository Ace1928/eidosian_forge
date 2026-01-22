from itertools import product
from typing import Tuple as tTuple
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (Function, ArgumentIndexError, expand_log,
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I, ImaginaryUnit
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint
def _eval_expand_log(self, deep=True, **hints):
    from sympy.concrete import Sum, Product
    force = hints.get('force', False)
    factor = hints.get('factor', False)
    if len(self.args) == 2:
        return expand_log(self.func(*self.args), deep=deep, force=force)
    arg = self.args[0]
    if arg.is_Integer:
        p = perfect_power(arg)
        logarg = None
        coeff = 1
        if p is not False:
            arg, coeff = p
            logarg = self.func(arg)
        if factor:
            p = factorint(arg)
            if arg not in p.keys():
                logarg = sum((n * log(val) for val, n in p.items()))
        if logarg is not None:
            return coeff * logarg
    elif arg.is_Rational:
        return log(arg.p) - log(arg.q)
    elif arg.is_Mul:
        expr = []
        nonpos = []
        for x in arg.args:
            if force or x.is_positive or x.is_polar:
                a = self.func(x)
                if isinstance(a, log):
                    expr.append(self.func(x)._eval_expand_log(**hints))
                else:
                    expr.append(a)
            elif x.is_negative:
                a = self.func(-x)
                expr.append(a)
                nonpos.append(S.NegativeOne)
            else:
                nonpos.append(x)
        return Add(*expr) + log(Mul(*nonpos))
    elif arg.is_Pow or isinstance(arg, exp):
        if force or (arg.exp.is_extended_real and (arg.base.is_positive or ((arg.exp + 1).is_positive and (arg.exp - 1).is_nonpositive))) or arg.base.is_polar:
            b = arg.base
            e = arg.exp
            a = self.func(b)
            if isinstance(a, log):
                return unpolarify(e) * a._eval_expand_log(**hints)
            else:
                return unpolarify(e) * a
    elif isinstance(arg, Product):
        if force or arg.function.is_positive:
            return Sum(log(arg.function), *arg.limits)
    return self.func(arg)