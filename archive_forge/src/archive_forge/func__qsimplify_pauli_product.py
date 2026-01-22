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
def _qsimplify_pauli_product(a, b):
    """
    Internal helper function for simplifying products of Pauli operators.
    """
    if not (isinstance(a, SigmaOpBase) and isinstance(b, SigmaOpBase)):
        return Mul(a, b)
    if a.name != b.name:
        if a.name < b.name:
            return Mul(a, b)
        else:
            return Mul(b, a)
    elif isinstance(a, SigmaX):
        if isinstance(b, SigmaX):
            return S.One
        if isinstance(b, SigmaY):
            return I * SigmaZ(a.name)
        if isinstance(b, SigmaZ):
            return -I * SigmaY(a.name)
        if isinstance(b, SigmaMinus):
            return S.Half + SigmaZ(a.name) / 2
        if isinstance(b, SigmaPlus):
            return S.Half - SigmaZ(a.name) / 2
    elif isinstance(a, SigmaY):
        if isinstance(b, SigmaX):
            return -I * SigmaZ(a.name)
        if isinstance(b, SigmaY):
            return S.One
        if isinstance(b, SigmaZ):
            return I * SigmaX(a.name)
        if isinstance(b, SigmaMinus):
            return -I * (S.One + SigmaZ(a.name)) / 2
        if isinstance(b, SigmaPlus):
            return I * (S.One - SigmaZ(a.name)) / 2
    elif isinstance(a, SigmaZ):
        if isinstance(b, SigmaX):
            return I * SigmaY(a.name)
        if isinstance(b, SigmaY):
            return -I * SigmaX(a.name)
        if isinstance(b, SigmaZ):
            return S.One
        if isinstance(b, SigmaMinus):
            return -SigmaMinus(a.name)
        if isinstance(b, SigmaPlus):
            return SigmaPlus(a.name)
    elif isinstance(a, SigmaMinus):
        if isinstance(b, SigmaX):
            return (S.One - SigmaZ(a.name)) / 2
        if isinstance(b, SigmaY):
            return -I * (S.One - SigmaZ(a.name)) / 2
        if isinstance(b, SigmaZ):
            return SigmaMinus(b.name)
        if isinstance(b, SigmaMinus):
            return S.Zero
        if isinstance(b, SigmaPlus):
            return S.Half - SigmaZ(a.name) / 2
    elif isinstance(a, SigmaPlus):
        if isinstance(b, SigmaX):
            return (S.One + SigmaZ(a.name)) / 2
        if isinstance(b, SigmaY):
            return I * (S.One + SigmaZ(a.name)) / 2
        if isinstance(b, SigmaZ):
            return -SigmaPlus(a.name)
        if isinstance(b, SigmaMinus):
            return (S.One + SigmaZ(a.name)) / 2
        if isinstance(b, SigmaPlus):
            return S.Zero
    else:
        return a * b