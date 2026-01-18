from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.exceptions import CircuitError
from ..boolean_logic import OR
from ..blueprintcircuit import BlueprintCircuit
If not already built, build the circuit.