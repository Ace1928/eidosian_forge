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
def _synth_su4(self, su4_mat, decomposer2q, preferred_direction, approximation_degree):
    approximate = not approximation_degree == 1.0
    synth_circ = decomposer2q(su4_mat, approximate=approximate)
    synth_direction = None
    for inst in synth_circ:
        if inst.operation.num_qubits == 2:
            synth_direction = [synth_circ.find_bit(q).index for q in inst.qubits]
    if preferred_direction and synth_direction != preferred_direction:
        su4_mat_mm = deepcopy(su4_mat)
        su4_mat_mm[[1, 2]] = su4_mat_mm[[2, 1]]
        su4_mat_mm[:, [1, 2]] = su4_mat_mm[:, [2, 1]]
        synth_circ = decomposer2q(su4_mat_mm, approximate=approximate).reverse_bits()
    return synth_circ