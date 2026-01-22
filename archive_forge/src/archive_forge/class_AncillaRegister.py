import itertools
from qiskit.circuit.exceptions import CircuitError
from .register import Register
from .bit import Bit
class AncillaRegister(QuantumRegister):
    """Implement an ancilla register."""
    instances_counter = itertools.count()
    prefix = 'a'
    bit_type = AncillaQubit