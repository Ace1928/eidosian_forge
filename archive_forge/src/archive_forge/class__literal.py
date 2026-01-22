from sympy.codegen.ast import (
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable
class _literal(Float):
    _token = None
    _decimals = None

    def _fcode(self, printer, *args, **kwargs):
        mantissa, sgnd_ex = ('%.{}e'.format(self._decimals) % self).split('e')
        mantissa = mantissa.strip('0').rstrip('.')
        ex_sgn, ex_num = (sgnd_ex[0], sgnd_ex[1:].lstrip('0'))
        ex_sgn = '' if ex_sgn == '+' else ex_sgn
        return (mantissa or '0') + self._token + ex_sgn + (ex_num or '0')