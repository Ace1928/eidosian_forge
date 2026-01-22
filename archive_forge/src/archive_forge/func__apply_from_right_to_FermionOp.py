from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, Ket, Bra
from sympy.functions.special.tensor_functions import KroneckerDelta
def _apply_from_right_to_FermionOp(self, op, **options):
    if op.is_annihilation:
        if self.n == 1:
            return FermionFockKet(0)
        else:
            return S.Zero
    elif self.n == 0:
        return FermionFockKet(1)
    else:
        return S.Zero