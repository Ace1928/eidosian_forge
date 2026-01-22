from __future__ import annotations
import logging
import numpy as np
from qiskit.circuit import Gate, ParameterExpression, Qubit
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.standard_gates import IGate, UGate, U3Gate
from qiskit.circuit.reset import Reset
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGInNode, DAGOpNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.target import Target
from .base_padding import BasePadding
def _constrained_length(values):
    return self._alignment * np.floor(values / self._alignment)