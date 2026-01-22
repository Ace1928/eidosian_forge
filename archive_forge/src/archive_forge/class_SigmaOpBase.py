from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum import Operator, Ket, Bra
from sympy.physics.quantum import ComplexSpace
from sympy.matrices import Matrix
from sympy.functions.special.tensor_functions import KroneckerDelta
class SigmaOpBase(Operator):
    """Pauli sigma operator, base class"""

    @property
    def name(self):
        return self.args[0]

    @property
    def use_name(self):
        return bool(self.args[0]) is not False

    @classmethod
    def default_args(self):
        return (False,)

    def __new__(cls, *args, **hints):
        return Operator.__new__(cls, *args, **hints)

    def _eval_commutator_BosonOp(self, other, **hints):
        return S.Zero