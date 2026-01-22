from sympy.core.expr import Expr
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.functions import sqrt
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError, QExpr
from sympy.matrices import eye
from sympy.physics.quantum.tensorproduct import matrix_tensor_product
from sympy.physics.quantum.gate import (
class RkGate(OneQubitGate):
    """This is the R_k gate of the QTF."""
    gate_name = 'Rk'
    gate_name_latex = 'R'

    def __new__(cls, *args):
        if len(args) != 2:
            raise QuantumError('Rk gates only take two arguments, got: %r' % args)
        target = args[0]
        k = args[1]
        if k == 1:
            return ZGate(target)
        elif k == 2:
            return PhaseGate(target)
        elif k == 3:
            return TGate(target)
        args = cls._eval_args(args)
        inst = Expr.__new__(cls, *args)
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _eval_args(cls, args):
        return QExpr._eval_args(args)

    @property
    def k(self):
        return self.label[1]

    @property
    def targets(self):
        return self.label[:1]

    @property
    def gate_name_plot(self):
        return '$%s_%s$' % (self.gate_name_latex, str(self.k))

    def get_target_matrix(self, format='sympy'):
        if format == 'sympy':
            return Matrix([[1, 0], [0, exp(Integer(2) * pi * I / Integer(2) ** self.k)]])
        raise NotImplementedError('Invalid format for the R_k gate: %r' % format)