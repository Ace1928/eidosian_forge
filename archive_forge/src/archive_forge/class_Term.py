from .add import Add
from .mul import Mul, _keep_coeff
from .power import Pow
from .basic import Basic
from .expr import Expr
from .function import expand_power_exp
from .sympify import sympify
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .coreerrors import NonCommutativeExpression
from .containers import Tuple, Dict
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import (common_prefix, common_suffix,
from collections import defaultdict
from typing import Tuple as tTuple
class Term:
    """Efficient representation of ``coeff*(numer/denom)``. """
    __slots__ = ('coeff', 'numer', 'denom')

    def __init__(self, term, numer=None, denom=None):
        if numer is None and denom is None:
            if not term.is_commutative:
                raise NonCommutativeExpression('commutative expression expected')
            coeff, factors = term.as_coeff_mul()
            numer, denom = (defaultdict(int), defaultdict(int))
            for factor in factors:
                base, exp = decompose_power(factor)
                if base.is_Add:
                    cont, base = base.primitive()
                    coeff *= cont ** exp
                if exp > 0:
                    numer[base] += exp
                else:
                    denom[base] += -exp
            numer = Factors(numer)
            denom = Factors(denom)
        else:
            coeff = term
            if numer is None:
                numer = Factors()
            if denom is None:
                denom = Factors()
        self.coeff = coeff
        self.numer = numer
        self.denom = denom

    def __hash__(self):
        return hash((self.coeff, self.numer, self.denom))

    def __repr__(self):
        return 'Term(%s, %s, %s)' % (self.coeff, self.numer, self.denom)

    def as_expr(self):
        return self.coeff * (self.numer.as_expr() / self.denom.as_expr())

    def mul(self, other):
        coeff = self.coeff * other.coeff
        numer = self.numer.mul(other.numer)
        denom = self.denom.mul(other.denom)
        numer, denom = numer.normal(denom)
        return Term(coeff, numer, denom)

    def inv(self):
        return Term(1 / self.coeff, self.denom, self.numer)

    def quo(self, other):
        return self.mul(other.inv())

    def pow(self, other):
        if other < 0:
            return self.inv().pow(-other)
        else:
            return Term(self.coeff ** other, self.numer.pow(other), self.denom.pow(other))

    def gcd(self, other):
        return Term(self.coeff.gcd(other.coeff), self.numer.gcd(other.numer), self.denom.gcd(other.denom))

    def lcm(self, other):
        return Term(self.coeff.lcm(other.coeff), self.numer.lcm(other.numer), self.denom.lcm(other.denom))

    def __mul__(self, other):
        if isinstance(other, Term):
            return self.mul(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Term):
            return self.quo(other)
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, SYMPY_INTS):
            return self.pow(other)
        else:
            return NotImplemented

    def __eq__(self, other):
        return self.coeff == other.coeff and self.numer == other.numer and (self.denom == other.denom)

    def __ne__(self, other):
        return not self == other