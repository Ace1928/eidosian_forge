from __future__ import annotations
import heapq
import math
from operator import itemgetter
from typing import Callable
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXXGate, RZXGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.one_qubit.one_qubit_decompose import ONE_QUBIT_EULER_BASIS_GATES
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from .circuits import apply_reflection, apply_shift, canonical_xx_circuit
from .utilities import EPSILON
from .polytopes import XXPolytope
@staticmethod
def _default_embodiment(strength):
    """
        If the user does not provide a custom implementation of XX(strength), then this routine
        defines a default implementation using RZX.
        """
    xx_circuit = QuantumCircuit(2)
    xx_circuit.h(0)
    xx_circuit.rzx(strength, 0, 1)
    xx_circuit.h(0)
    return xx_circuit