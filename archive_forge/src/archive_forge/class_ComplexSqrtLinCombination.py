import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
class ComplexSqrtLinCombination:
    """
    A pair (real, imag) of SqrtLinCombinations representing the complex number
    real + imag * I. Supports ``real()``, ``imag()``, ``+``, ``-``, ``*``, ``/``,
    ``abs``, ``conjugate()`` and ``==``.
    """

    def __init__(self, real, imag=0, embed_cache=None):
        if isinstance(real, SqrtLinCombination):
            self._real = real
        else:
            self._real = SqrtLinCombination(real, embed_cache=embed_cache)
        if isinstance(imag, SqrtLinCombination):
            self._imag = imag
        else:
            self._imag = SqrtLinCombination(imag, embed_cache=embed_cache)

    def __repr__(self):
        return 'ComplexSqrtLinCombination(%r, %r)' % (self._real, self._imag)

    def real(self):
        """
        Real part.
        """
        return self._real

    def imag(self):
        """
        Imaginary part.
        """
        return self._imag

    def __abs__(self):
        """
        Absolute value.
        """
        return sqrt(self._real * self._real + self._imag * self._imag)

    def __add__(self, other):
        if not isinstance(other, ComplexSqrtLinCombination):
            return self + ComplexSqrtLinCombination(other)
        return ComplexSqrtLinCombination(self._real + other._real, self._imag + other._imag)

    def __neg__(self):
        return ComplexSqrtLinCombination(-self._real, -self._imag)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if not isinstance(other, ComplexSqrtLinCombination):
            return self * ComplexSqrtLinCombination(other)
        return ComplexSqrtLinCombination(self._real * other._real - self._imag * other._imag, self._real * other._imag + self._imag * other._real)

    def __div__(self, other):
        if not isinstance(other, ComplexSqrtLinCombination):
            return self / ComplexSqrtLinCombination(other)
        num = 1 / (other._real * other._real + other._imag * other._imag)
        return ComplexSqrtLinCombination((self._real * other._real + self._imag * other._imag) * num, (other._real * self._imag - self._real * other._imag) * num)

    def __truediv__(self, other):
        return self.__div__(other)

    def conjugate(self):
        return ComplexSqrtLinCombination(self._real, -self._imag)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rdiv__(self, other):
        return ComplexSqrtLinCombination(other) / self

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __eq__(self, other):
        if not isinstance(other, ComplexSqrtLinCombination):
            return self == ComplexSqrtLinCombination(other)
        return self._real == other._real and self._imag == other._imag

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        raise TypeError('No order on complex numbers.')

    def __le__(self, other):
        raise TypeError('No order on complex numbers.')

    def __gt__(self, other):
        raise TypeError('No order on complex numbers.')

    def __ge__(self, other):
        raise TypeError('No order on complex numbers.')

    def _complex_mpfi_(self, CIF):
        """
        Convert to complex interval in given ComplexIntervalField instance.
        """
        RIF = CIF(0).real().parent()
        return CIF(RIF(self._real), RIF(self._imag))