from functools import reduce
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
class ComplexSpace(HilbertSpace):
    """Finite dimensional Hilbert space of complex vectors.

    The elements of this Hilbert space are n-dimensional complex valued
    vectors with the usual inner product that takes the complex conjugate
    of the vector on the right.

    A classic example of this type of Hilbert space is spin-1/2, which is
    ``ComplexSpace(2)``. Generalizing to spin-s, the space is
    ``ComplexSpace(2*s+1)``.  Quantum computing with N qubits is done with the
    direct product space ``ComplexSpace(2)**N``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.hilbert import ComplexSpace
    >>> c1 = ComplexSpace(2)
    >>> c1
    C(2)
    >>> c1.dimension
    2

    >>> n = symbols('n')
    >>> c2 = ComplexSpace(n)
    >>> c2
    C(n)
    >>> c2.dimension
    n

    """

    def __new__(cls, dimension):
        dimension = sympify(dimension)
        r = cls.eval(dimension)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, dimension)
        return obj

    @classmethod
    def eval(cls, dimension):
        if len(dimension.atoms()) == 1:
            if not (dimension.is_Integer and dimension > 0 or dimension is S.Infinity or dimension.is_Symbol):
                raise TypeError('The dimension of a ComplexSpace can onlybe a positive integer, oo, or a Symbol: %r' % dimension)
        else:
            for dim in dimension.atoms():
                if not (dim.is_Integer or dim is S.Infinity or dim.is_Symbol):
                    raise TypeError('The dimension of a ComplexSpace can only contain integers, oo, or a Symbol: %r' % dim)

    @property
    def dimension(self):
        return self.args[0]

    def _sympyrepr(self, printer, *args):
        return '%s(%s)' % (self.__class__.__name__, printer._print(self.dimension, *args))

    def _sympystr(self, printer, *args):
        return 'C(%s)' % printer._print(self.dimension, *args)

    def _pretty(self, printer, *args):
        ustr = 'C'
        pform_exp = printer._print(self.dimension, *args)
        pform_base = prettyForm(ustr)
        return pform_base ** pform_exp

    def _latex(self, printer, *args):
        return '\\mathcal{C}^{%s}' % printer._print(self.dimension, *args)