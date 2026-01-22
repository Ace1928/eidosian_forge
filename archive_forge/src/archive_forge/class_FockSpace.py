from functools import reduce
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
class FockSpace(HilbertSpace):
    """The Hilbert space for second quantization.

    Technically, this Hilbert space is a infinite direct sum of direct
    products of single particle Hilbert spaces [1]_. This is a mess, so we have
    a class to represent it directly.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import FockSpace
    >>> hs = FockSpace()
    >>> hs
    F
    >>> hs.dimension
    oo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fock_space
    """

    def __new__(cls):
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        return S.Infinity

    def _sympyrepr(self, printer, *args):
        return 'FockSpace()'

    def _sympystr(self, printer, *args):
        return 'F'

    def _pretty(self, printer, *args):
        ustr = 'F'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        return '\\mathcal{F}'