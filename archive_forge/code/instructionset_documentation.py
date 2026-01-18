from __future__ import annotations
from collections.abc import MutableSequence
from typing import Callable
from qiskit.circuit.exceptions import CircuitError
from .classicalregister import Clbit, ClassicalRegister
from .operation import Operation
from .quantumcircuitdata import CircuitInstruction
Legacy getter for the cargs components of an instruction set.  This does not support
        mutation.