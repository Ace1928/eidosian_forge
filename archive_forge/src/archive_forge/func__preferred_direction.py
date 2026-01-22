from __future__ import annotations
from math import pi, inf, isclose
from typing import Any
from copy import deepcopy
from itertools import product
from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.synthesis.two_qubit.xx_decompose import XXDecomposer, XXEmbodiments
from qiskit.synthesis.two_qubit.two_qubit_decompose import (
from qiskit.quantum_info import Operator
from qiskit.circuit import ControlFlowOp, Gate, Parameter
from qiskit.circuit.library.standard_gates import (
from qiskit.transpiler.passes.synthesis import plugin
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
from qiskit.providers.models import BackendProperties
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
def _preferred_direction(decomposer2q, qubits, natural_direction, coupling_map=None, gate_lengths=None, gate_errors=None):
    """
    `decomposer2q` decomposes an SU(4) over `qubits`. A user sets `natural_direction`
    to indicate whether they prefer synthesis in a hardware-native direction.
    If yes, we return the `preferred_direction` here. If no hardware direction is
    preferred, we raise an error (unless natural_direction is None).
    We infer this from `coupling_map`, `gate_lengths`, `gate_errors`.

    Returns [0, 1] if qubits are correct in the hardware-native direction.
    Returns [1, 0] if qubits must be flipped to match hardware-native direction.
    """
    qubits_tuple = tuple(qubits)
    reverse_tuple = qubits_tuple[::-1]
    preferred_direction = None
    if natural_direction in {None, True}:
        if coupling_map is not None:
            neighbors0 = coupling_map.neighbors(qubits[0])
            zero_one = qubits[1] in neighbors0
            neighbors1 = coupling_map.neighbors(qubits[1])
            one_zero = qubits[0] in neighbors1
            if zero_one and (not one_zero):
                preferred_direction = [0, 1]
            if one_zero and (not zero_one):
                preferred_direction = [1, 0]
        if preferred_direction is None and (gate_lengths or gate_errors):
            cost_0_1 = inf
            cost_1_0 = inf
            try:
                cost_0_1 = next((duration for gate, duration in gate_lengths.get(qubits_tuple, []) if gate == decomposer2q.gate))
            except StopIteration:
                pass
            try:
                cost_1_0 = next((duration for gate, duration in gate_lengths.get(reverse_tuple, []) if gate == decomposer2q.gate))
            except StopIteration:
                pass
            if not (cost_0_1 < inf or cost_1_0 < inf):
                try:
                    cost_0_1 = next((error for gate, error in gate_errors.get(qubits_tuple, []) if gate == decomposer2q.gate))
                except StopIteration:
                    pass
                try:
                    cost_1_0 = next((error for gate, error in gate_errors.get(reverse_tuple, []) if gate == decomposer2q.gate))
                except StopIteration:
                    pass
            if cost_0_1 < cost_1_0:
                preferred_direction = [0, 1]
            elif cost_1_0 < cost_0_1:
                preferred_direction = [1, 0]
    if natural_direction is True and preferred_direction is None:
        raise TranspilerError(f'No preferred direction of gate on qubits {qubits} could be determined from coupling map or gate lengths / gate errors.')
    return preferred_direction