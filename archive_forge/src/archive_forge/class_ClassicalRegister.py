import itertools
from qiskit.circuit.exceptions import CircuitError
from .register import Register
from .bit import Bit
class ClassicalRegister(Register):
    """Implement a classical register."""
    instances_counter = itertools.count()
    prefix = 'c'
    bit_type = Clbit