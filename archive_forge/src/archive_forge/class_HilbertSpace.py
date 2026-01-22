from functools import reduce
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
class HilbertSpace(Basic):
    """An abstract Hilbert space for quantum mechanics.

    In short, a Hilbert space is an abstract vector space that is complete
    with inner products defined [1]_.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import HilbertSpace
    >>> hs = HilbertSpace()
    >>> hs
    H

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space
    """

    def __new__(cls):
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        """Return the Hilbert dimension of the space."""
        raise NotImplementedError('This Hilbert space has no dimension.')

    def __add__(self, other):
        return DirectSumHilbertSpace(self, other)

    def __radd__(self, other):
        return DirectSumHilbertSpace(other, self)

    def __mul__(self, other):
        return TensorProductHilbertSpace(self, other)

    def __rmul__(self, other):
        return TensorProductHilbertSpace(other, self)

    def __pow__(self, other, mod=None):
        if mod is not None:
            raise ValueError('The third argument to __pow__ is not supported             for Hilbert spaces.')
        return TensorPowerHilbertSpace(self, other)

    def __contains__(self, other):
        """Is the operator or state in this Hilbert space.

        This is checked by comparing the classes of the Hilbert spaces, not
        the instances. This is to allow Hilbert Spaces with symbolic
        dimensions.
        """
        if other.hilbert_space.__class__ == self.__class__:
            return True
        else:
            return False

    def _sympystr(self, printer, *args):
        return 'H'

    def _pretty(self, printer, *args):
        ustr = 'H'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        return '\\mathcal{H}'