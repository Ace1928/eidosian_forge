from __future__ import annotations
import copy
import re
from numbers import Number
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.operator import Operator, BaseOperator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit._accelerate.pauli_expval import (
Update the current Statevector by applying an instruction.