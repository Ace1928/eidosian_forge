from __future__ import annotations
from collections.abc import MutableSequence
from typing import Callable
from qiskit.circuit.exceptions import CircuitError
from .classicalregister import Clbit, ClassicalRegister
from .operation import Operation
from .quantumcircuitdata import CircuitInstruction
def _instructions_iter(self):
    return (i if isinstance(i, CircuitInstruction) else i[0][i[1]] for i in self._instructions)