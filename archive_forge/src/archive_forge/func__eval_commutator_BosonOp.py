from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, FockSpace, Ket, Bra, IdentityOperator
from sympy.functions.special.tensor_functions import KroneckerDelta
def _eval_commutator_BosonOp(self, other, **hints):
    if self.name == other.name:
        if not self.is_annihilation and other.is_annihilation:
            return S.NegativeOne
    elif 'independent' in hints and hints['independent']:
        return S.Zero
    return None