from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, FockSpace, Ket, Bra, IdentityOperator
from sympy.functions.special.tensor_functions import KroneckerDelta
def _eval_innerproduct_BosonCoherentBra(self, bra, **hints):
    if self.alpha == bra.alpha:
        return S.One
    else:
        return exp(-(abs(self.alpha) ** 2 + abs(bra.alpha) ** 2 - 2 * conjugate(bra.alpha) * self.alpha) / 2)