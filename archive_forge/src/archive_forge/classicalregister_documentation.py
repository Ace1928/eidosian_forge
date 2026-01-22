import itertools
from qiskit.circuit.exceptions import CircuitError
from .register import Register
from .bit import Bit
Creates a classical bit.

        Args:
            register (ClassicalRegister): Optional. A classical register containing the bit.
            index (int): Optional. The index of the bit in its containing register.

        Raises:
            CircuitError: if the provided register is not a valid :class:`ClassicalRegister`
        