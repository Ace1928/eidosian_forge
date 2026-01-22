from sympy.core.numbers import (I, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.cartesian import X, Px
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.matrixutils import matrix_zeros
class SHOBra(SHOState, Bra):
    """A time-independent Bra in SHO.

    Inherits from SHOState and Bra.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket
        This is usually its quantum numbers or its symbol.

    Examples
    ========

    Bra's know about their associated ket:

        >>> from sympy.physics.quantum.sho1d import SHOBra

        >>> b = SHOBra('b')
        >>> b.dual
        |b>
        >>> b.dual_class()
        <class 'sympy.physics.quantum.sho1d.SHOKet'>

    Vector representation of a numerical state bra:

        >>> from sympy.physics.quantum.sho1d import SHOBra, NumberOp
        >>> from sympy.physics.quantum.represent import represent

        >>> b = SHOBra(3)
        >>> N = NumberOp('N')
        >>> represent(b, basis=N, ndim=4)
        Matrix([[0, 0, 0, 1]])

    """

    @classmethod
    def dual_class(self):
        return SHOKet

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        options['spmatrix'] = 'lil'
        vector = matrix_zeros(1, ndim_info, **options)
        if isinstance(self.n, Integer):
            if self.n >= ndim_info:
                return ValueError('N-Dimension too small')
            if format == 'scipy.sparse':
                vector[0, int(self.n)] = 1.0
                vector = vector.tocsr()
            elif format == 'numpy':
                vector[0, int(self.n)] = 1.0
            else:
                vector[0, self.n] = S.One
            return vector
        else:
            return ValueError('Not Numerical State')