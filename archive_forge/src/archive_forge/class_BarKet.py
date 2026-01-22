from sympy.core.numbers import (I, Integer)
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import Bra, Ket, StateBase
class BarKet(Ket, BarState):

    @classmethod
    def dual_class(self):
        return BarBra