import itertools
from qiskit.circuit.exceptions import CircuitError
from .register import Register
from .bit import Bit
class Clbit(Bit):
    """Implement a classical bit."""
    __slots__ = ()

    def __init__(self, register=None, index=None):
        """Creates a classical bit.

        Args:
            register (ClassicalRegister): Optional. A classical register containing the bit.
            index (int): Optional. The index of the bit in its containing register.

        Raises:
            CircuitError: if the provided register is not a valid :class:`ClassicalRegister`
        """
        if register is None or isinstance(register, ClassicalRegister):
            super().__init__(register, index)
        else:
            raise CircuitError('Clbit needs a ClassicalRegister and %s was provided' % type(register).__name__)