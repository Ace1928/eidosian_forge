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

        Fashions a circuit which (perhaps approximately) models the special unitary operation
        ``unitary``, using the circuit templates supplied at initialization as ``embodiments``.  The
        routine uses ``basis_fidelity`` to select the optimal circuit template, including when
        performing exact synthesis; the contents of ``basis_fidelity`` is a dictionary mapping
        interaction strengths (scaled so that :math:`CX = RZX(\pi/2)` corresponds to :math:`\pi/2`)
        to circuit fidelities.

        Args:
            unitary (Operator or ndarray): :math:`4 \times 4` unitary to synthesize.
            basis_fidelity (dict or float): Fidelity of basis gates. Can be either (1) a dictionary
                mapping ``XX`` angle values to fidelity at that angle; or (2) a single float ``f``,
                interpreted as ``{pi: f, pi/2: f/2, pi/3: f/3}``.
                If given, overrides the basis_fidelity given at init.
            approximate (bool): Approximates if basis fidelities are less than 1.0 .

        Returns:
            QuantumCircuit: Synthesized circuit.
        