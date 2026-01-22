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
def _best_decomposition(canonical_coordinate, available_strengths):
    """
        Finds the cheapest sequence of `available_strengths` which supports the best approximation
        to `canonical_coordinate`. Returns a dictionary with keys "cost", "point", and "operations".

        NOTE: `canonical_coordinate` is a positive canonical coordinate. `strengths` is a dictionary
              mapping the available strengths to their (infidelity) costs, with the strengths
              themselves normalized so that pi/2 represents CX = RZX(pi/2).
        """
    best_point, best_cost, best_sequence = ([0, 0, 0], 1.0, [])
    priority_queue = []
    heapq.heappush(priority_queue, (0, []))
    canonical_coordinate = np.array(canonical_coordinate)
    while True:
        if len(priority_queue) == 0:
            if len(available_strengths) == 0:
                raise QiskitError('Attempting to synthesize entangling gate with no controlled gates in basis set.')
            raise QiskitError('Unable to synthesize a 2q unitary with the supplied basis set.')
        sequence_cost, sequence = heapq.heappop(priority_queue)
        strength_polytope = XXPolytope.from_strengths(*[x / 2 for x in sequence])
        candidate_point = strength_polytope.nearest(canonical_coordinate)
        candidate_cost = sequence_cost + _average_infidelity(canonical_coordinate, candidate_point)
        if candidate_cost < best_cost:
            best_point, best_cost, best_sequence = (candidate_point, candidate_cost, sequence)
        if strength_polytope.member(canonical_coordinate):
            break
        for strength, extra_cost in available_strengths.items():
            if len(sequence) == 0 or strength <= sequence[-1]:
                heapq.heappush(priority_queue, (sequence_cost + extra_cost, sequence + [strength]))
    return {'point': best_point, 'cost': best_cost, 'sequence': best_sequence}