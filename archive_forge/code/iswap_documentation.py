from typing import Optional
import numpy as np
from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array
from .xx_plus_yy import XXPlusYYGate
Raise gate to a power.