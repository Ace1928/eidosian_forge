import sympy
from sympy import S
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b)
    """
    nargs = (2,)
    precedence = 50
    is_real = True

    @property
    def base(self):
        return self.args[0]

    @property
    def divisor(self):
        return self.args[1]

    def _sympystr(self, printer):
        base = printer.parenthesize(self.base, self.precedence)
        divisor = printer.parenthesize(self.divisor, self.precedence)
        return f'({base}//{divisor})'

    def _eval_is_real(self):
        return fuzzy_or([self.base.is_real, self.divisor.is_real])

    def _eval_is_integer(self):
        return fuzzy_and([self.base.is_integer, self.divisor.is_integer])

    @classmethod
    def eval(cls, base, divisor):

        def check_supported_type(x):
            if x.is_integer is False and x.is_real is False and x.is_complex or x.is_Boolean:
                raise TypeError(f"unsupported operand type(s) for //: '{type(base).__name__}' and '{type(divisor).__name__}', expected integer or real")
        check_supported_type(base)
        check_supported_type(divisor)
        if divisor.is_zero:
            raise ZeroDivisionError('division by zero')
        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and divisor == 1:
            return base
        if base.is_real and divisor == 1:
            return sympy.floor(base)
        if base.is_integer and divisor == -1:
            return sympy.Mul(base, -1)
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return base // divisor
        if isinstance(base, (sympy.Integer, sympy.Float)) and isinstance(divisor, (sympy.Integer, sympy.Float)):
            return sympy.floor(base / divisor)
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)
        if isinstance(divisor, sympy.Rational) and divisor.p == 1:
            return sympy.floor(base * divisor.q)
        if isinstance(base, sympy.Add):
            for a in base.args:
                gcd = sympy.gcd(a, divisor)
                if gcd == divisor:
                    return FloorDiv(base - a, divisor) + a / gcd
        try:
            gcd = sympy.gcd(base, divisor)
            if gcd != 1:
                return FloorDiv(sympy.simplify(base / gcd), sympy.simplify(divisor / gcd))
        except sympy.PolynomialError:
            pass