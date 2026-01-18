from __future__ import annotations
import copy
from typing import TYPE_CHECKING
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor, _to_superop
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
Update the current Operator by apply an instruction.