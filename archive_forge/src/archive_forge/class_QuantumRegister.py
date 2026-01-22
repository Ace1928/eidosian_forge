import itertools
from qiskit.circuit.exceptions import CircuitError
from .register import Register
from .bit import Bit
class QuantumRegister(Register):
    """Implement a quantum register."""
    instances_counter = itertools.count()
    prefix = 'q'
    bit_type = Qubit