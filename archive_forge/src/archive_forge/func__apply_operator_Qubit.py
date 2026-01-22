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
def _apply_operator_Qubit(self, qubits, **options):
    """
        qubits: a set of qubits (Qubit)
        Returns: quantum object (quantum expression - QExpr)
        """
    if qubits.nqubits != self.nqubits:
        raise QuantumError('WGate operates on %r qubits, got: %r' % (self.nqubits, qubits.nqubits))
    basis_states = superposition_basis(self.nqubits)
    change_to_basis = 2 / sqrt(2 ** self.nqubits) * basis_states
    return change_to_basis - qubits