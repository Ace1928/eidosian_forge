from collections.abc import MutableSequence
import qiskit._accelerate.quantum_circuit
from .exceptions import CircuitError
from .instruction import Instruction
from .operation import Operation
Returns a shallow copy of instruction list.