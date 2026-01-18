from __future__ import annotations
import copy
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_ptm
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.base_operator import BaseOperator
Return the shape for bipartite matrix