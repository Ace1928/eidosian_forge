from sympy.core import Basic, Integer
import operator
class OmegaPower(Basic):
    """
    Represents ordinal exponential and multiplication terms one of the
    building blocks of the :class:`Ordinal` class.
    In ``OmegaPower(a, b)``, ``a`` represents exponent and ``b`` represents multiplicity.
    """

    def __new__(cls, a, b):
        if isinstance(b, int):
            b = Integer(b)
        if not isinstance(b, Integer) or b <= 0:
            raise TypeError('multiplicity must be a positive integer')
        if not isinstance(a, Ordinal):
            a = Ordinal.convert(a)
        return Basic.__new__(cls, a, b)

    @property
    def exp(self):
        return self.args[0]

    @property
    def mult(self):
        return self.args[1]

    def _compare_term(self, other, op):
        if self.exp == other.exp:
            return op(self.mult, other.mult)
        else:
            return op(self.exp, other.exp)

    def __eq__(self, other):
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self.args == other.args

    def __hash__(self):
        return Basic.__hash__(self)

    def __lt__(self, other):
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self._compare_term(other, operator.lt)