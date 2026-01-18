from __future__ import annotations
import math
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
Calculate the new rotation angles according to Shende's decomposition.