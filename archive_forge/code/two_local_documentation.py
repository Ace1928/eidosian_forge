from __future__ import annotations
import typing
from collections.abc import Callable, Sequence
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from .n_local import NLocal
from ..standard_gates import (
Overloading to handle the special case of 1 qubit where the entanglement are ignored.