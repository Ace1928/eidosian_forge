from __future__ import annotations
import copy
from typing import Optional, Union
from qiskit.circuit.exceptions import CircuitError
from .quantumcircuit import QuantumCircuit
from .gate import Gate
from .quantumregister import QuantumRegister
from ._utils import _ctrl_state_to_int
Invert this gate by calling inverse on the base gate.