from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply
class JzKetCoupled(CoupledSpinState, Ket):
    """Coupled eigenket of Jz

    Spin state that is an eigenket of Jz which represents the coupling of
    separate spin spaces.

    The arguments for creating instances of JzKetCoupled are ``j``, ``m``,
    ``jn`` and an optional ``jcoupling`` argument. The ``j`` and ``m`` options
    are the total angular momentum quantum numbers, as used for normal states
    (e.g. JzKet).

    The other required parameter in ``jn``, which is a tuple defining the `j_n`
    angular momentum quantum numbers of the product spaces. So for example, if
    a state represented the coupling of the product basis state
    `\\left|j_1,m_1\\right\\rangle\\times\\left|j_2,m_2\\right\\rangle`, the ``jn``
    for this state would be ``(j1,j2)``.

    The final option is ``jcoupling``, which is used to define how the spaces
    specified by ``jn`` are coupled, which includes both the order these spaces
    are coupled together and the quantum numbers that arise from these
    couplings. The ``jcoupling`` parameter itself is a list of lists, such that
    each of the sublists defines a single coupling between the spin spaces. If
    there are N coupled angular momentum spaces, that is ``jn`` has N elements,
    then there must be N-1 sublists. Each of these sublists making up the
    ``jcoupling`` parameter have length 3. The first two elements are the
    indices of the product spaces that are considered to be coupled together.
    For example, if we want to couple `j_1` and `j_4`, the indices would be 1
    and 4. If a state has already been coupled, it is referenced by the
    smallest index that is coupled, so if `j_2` and `j_4` has already been
    coupled to some `j_{24}`, then this value can be coupled by referencing it
    with index 2. The final element of the sublist is the quantum number of the
    coupled state. So putting everything together, into a valid sublist for
    ``jcoupling``, if `j_1` and `j_2` are coupled to an angular momentum space
    with quantum number `j_{12}` with the value ``j12``, the sublist would be
    ``(1,2,j12)``, N-1 of these sublists are used in the list for
    ``jcoupling``.

    Note the ``jcoupling`` parameter is optional, if it is not specified, the
    default coupling is taken. This default value is to coupled the spaces in
    order and take the quantum number of the coupling to be the maximum value.
    For example, if the spin spaces are `j_1`, `j_2`, `j_3`, `j_4`, then the
    default coupling couples `j_1` and `j_2` to `j_{12}=j_1+j_2`, then,
    `j_{12}` and `j_3` are coupled to `j_{123}=j_{12}+j_3`, and finally
    `j_{123}` and `j_4` to `j=j_{123}+j_4`. The jcoupling value that would
    correspond to this is:

        ``((1,2,j1+j2),(1,3,j1+j2+j3))``

    Parameters
    ==========

    args : tuple
        The arguments that must be passed are ``j``, ``m``, ``jn``, and
        ``jcoupling``. The ``j`` value is the total angular momentum. The ``m``
        value is the eigenvalue of the Jz spin operator. The ``jn`` list are
        the j values of argular momentum spaces coupled together. The
        ``jcoupling`` parameter is an optional parameter defining how the spaces
        are coupled together. See the above description for how these coupling
        parameters are defined.

    Examples
    ========

    Defining simple spin states, both numerical and symbolic:

        >>> from sympy.physics.quantum.spin import JzKetCoupled
        >>> from sympy import symbols
        >>> JzKetCoupled(1, 0, (1, 1))
        |1,0,j1=1,j2=1>
        >>> j, m, j1, j2 = symbols('j m j1 j2')
        >>> JzKetCoupled(j, m, (j1, j2))
        |j,m,j1=j1,j2=j2>

    Defining coupled spin states for more than 2 coupled spaces with various
    coupling parameters:

        >>> JzKetCoupled(2, 1, (1, 1, 1))
        |2,1,j1=1,j2=1,j3=1,j(1,2)=2>
        >>> JzKetCoupled(2, 1, (1, 1, 1), ((1,2,2),(1,3,2)) )
        |2,1,j1=1,j2=1,j3=1,j(1,2)=2>
        >>> JzKetCoupled(2, 1, (1, 1, 1), ((2,3,1),(1,2,2)) )
        |2,1,j1=1,j2=1,j3=1,j(2,3)=1>

    Rewriting the JzKetCoupled in terms of eigenkets of the Jx operator:
    Note: that the resulting eigenstates are JxKetCoupled

        >>> JzKetCoupled(1,1,(1,1)).rewrite("Jx")
        |1,-1,j1=1,j2=1>/2 - sqrt(2)*|1,0,j1=1,j2=1>/2 + |1,1,j1=1,j2=1>/2

    The rewrite method can be used to convert a coupled state to an uncoupled
    state. This is done by passing coupled=False to the rewrite function:

        >>> JzKetCoupled(1, 0, (1, 1)).rewrite('Jz', coupled=False)
        -sqrt(2)*|1,-1>x|1,1>/2 + sqrt(2)*|1,1>x|1,-1>/2

    Get the vector representation of a state in terms of the basis elements
    of the Jx operator:

        >>> from sympy.physics.quantum.represent import represent
        >>> from sympy.physics.quantum.spin import Jx
        >>> from sympy import S
        >>> represent(JzKetCoupled(1,-1,(S(1)/2,S(1)/2)), basis=Jx)
        Matrix([
        [        0],
        [      1/2],
        [sqrt(2)/2],
        [      1/2]])

    See Also
    ========

    JzKet: Normal spin eigenstates
    uncouple: Uncoupling of coupling spin states
    couple: Coupling of uncoupled spin states

    """

    @classmethod
    def dual_class(self):
        return JzBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JzKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(beta=pi * Rational(3, 2), **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi * Rational(3, 2), beta=pi / 2, gamma=pi / 2, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(**options)