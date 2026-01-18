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
@property
def use_name(self):
    return bool(self.args[0]) is not False