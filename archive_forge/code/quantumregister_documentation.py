import itertools
from qiskit.circuit.exceptions import CircuitError
from .register import Register
from .bit import Bit
Creates a qubit.

        Args:
            register (QuantumRegister): Optional. A quantum register containing the bit.
            index (int): Optional. The index of the bit in its containing register.

        Raises:
            CircuitError: if the provided register is not a valid :class:`QuantumRegister`
        