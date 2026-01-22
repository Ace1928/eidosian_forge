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
class J2Op(SpinOpBase, HermitianOperator):
    """The J^2 operator."""
    _coord = '2'

    def _eval_commutator_JxOp(self, other):
        return S.Zero

    def _eval_commutator_JyOp(self, other):
        return S.Zero

    def _eval_commutator_JzOp(self, other):
        return S.Zero

    def _eval_commutator_JplusOp(self, other):
        return S.Zero

    def _eval_commutator_JminusOp(self, other):
        return S.Zero

    def _apply_operator_JxKet(self, ket, **options):
        j = ket.j
        return hbar ** 2 * j * (j + 1) * ket

    def _apply_operator_JxKetCoupled(self, ket, **options):
        j = ket.j
        return hbar ** 2 * j * (j + 1) * ket

    def _apply_operator_JyKet(self, ket, **options):
        j = ket.j
        return hbar ** 2 * j * (j + 1) * ket

    def _apply_operator_JyKetCoupled(self, ket, **options):
        j = ket.j
        return hbar ** 2 * j * (j + 1) * ket

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        return hbar ** 2 * j * (j + 1) * ket

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        return hbar ** 2 * j * (j + 1) * ket

    def matrix_element(self, j, m, jp, mp):
        result = hbar ** 2 * j * (j + 1)
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _print_contents_pretty(self, printer, *args):
        a = prettyForm(str(self.name))
        b = prettyForm('2')
        return a ** b

    def _print_contents_latex(self, printer, *args):
        return '%s^2' % str(self.name)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) ** 2 + JyOp(args[0]) ** 2 + JzOp(args[0]) ** 2

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        a = args[0]
        return JzOp(a) ** 2 + S.Half * (JplusOp(a) * JminusOp(a) + JminusOp(a) * JplusOp(a))