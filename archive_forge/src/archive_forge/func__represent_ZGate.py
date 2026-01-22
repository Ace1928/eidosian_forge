from sympy.core.numbers import pi
from sympy.core.sympify import sympify
from sympy.core.basic import Atom
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import UnitaryOperator
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import IntQubit
def _represent_ZGate(self, basis, **options):
    """
        Represent the OracleGate in the computational basis.
        """
    nbasis = 2 ** self.nqubits
    matrixOracle = eye(nbasis)
    for i in range(nbasis):
        if self.search_function(IntQubit(i, nqubits=self.nqubits)):
            matrixOracle[i, i] = NegativeOne()
    return matrixOracle