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
class RaisingOp(SHOOp):
    """The Raising Operator or a^dagger.

    When a^dagger acts on a state it raises the state up by one. Taking
    the adjoint of a^dagger returns 'a', the Lowering Operator. a^dagger
    can be rewritten in terms of position and momentum. We can represent
    a^dagger as a matrix, which will be its default basis.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator.

    Examples
    ========

    Create a Raising Operator and rewrite it in terms of position and
    momentum, and show that taking its adjoint returns 'a':

        >>> from sympy.physics.quantum.sho1d import RaisingOp
        >>> from sympy.physics.quantum import Dagger

        >>> ad = RaisingOp('a')
        >>> ad.rewrite('xp').doit()
        sqrt(2)*(m*omega*X - I*Px)/(2*sqrt(hbar)*sqrt(m*omega))

        >>> Dagger(ad)
        a

    Taking the commutator of a^dagger with other Operators:

        >>> from sympy.physics.quantum import Commutator
        >>> from sympy.physics.quantum.sho1d import RaisingOp, LoweringOp
        >>> from sympy.physics.quantum.sho1d import NumberOp

        >>> ad = RaisingOp('a')
        >>> a = LoweringOp('a')
        >>> N = NumberOp('N')
        >>> Commutator(ad, a).doit()
        -1
        >>> Commutator(ad, N).doit()
        -RaisingOp(a)

    Apply a^dagger to a state:

        >>> from sympy.physics.quantum import qapply
        >>> from sympy.physics.quantum.sho1d import RaisingOp, SHOKet

        >>> ad = RaisingOp('a')
        >>> k = SHOKet('k')
        >>> qapply(ad*k)
        sqrt(k + 1)*|k + 1>

    Matrix Representation

        >>> from sympy.physics.quantum.sho1d import RaisingOp
        >>> from sympy.physics.quantum.represent import represent
        >>> ad = RaisingOp('a')
        >>> represent(ad, basis=N, ndim=4, format='sympy')
        Matrix([
        [0,       0,       0, 0],
        [1,       0,       0, 0],
        [0, sqrt(2),       0, 0],
        [0,       0, sqrt(3), 0]])

    """

    def _eval_rewrite_as_xp(self, *args, **kwargs):
        return S.One / sqrt(Integer(2) * hbar * m * omega) * (S.NegativeOne * I * Px + m * omega * X)

    def _eval_adjoint(self):
        return LoweringOp(*self.args)

    def _eval_commutator_LoweringOp(self, other):
        return S.NegativeOne

    def _eval_commutator_NumberOp(self, other):
        return S.NegativeOne * self

    def _apply_operator_SHOKet(self, ket, **options):
        temp = ket.n + S.One
        return sqrt(temp) * SHOKet(temp)

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        raise NotImplementedError('Position representation is not implemented')

    def _represent_NumberOp(self, basis, **options):
        ndim_info = options.get('ndim', 4)
        format = options.get('format', 'sympy')
        matrix = matrix_zeros(ndim_info, ndim_info, **options)
        for i in range(ndim_info - 1):
            value = sqrt(i + 1)
            if format == 'scipy.sparse':
                value = float(value)
            matrix[i + 1, i] = value
        if format == 'scipy.sparse':
            matrix = matrix.tocsr()
        return matrix

    def _print_contents(self, printer, *args):
        arg0 = printer._print(self.args[0], *args)
        return '%s(%s)' % (self.__class__.__name__, arg0)

    def _print_contents_pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        pform = pform ** prettyForm('†')
        return pform

    def _print_contents_latex(self, printer, *args):
        arg = printer._print(self.args[0])
        return '%s^{\\dagger}' % arg