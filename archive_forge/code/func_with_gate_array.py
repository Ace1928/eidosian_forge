import numpy
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from .parametervector import ParameterVectorElement
def with_gate_array(base_array):
    """Class decorator that adds an ``__array__`` method to a :class:`.Gate` instance that returns a
    singleton nonwritable view onto the complex matrix described by ``base_array``."""
    nonwritable = numpy.array(base_array, dtype=numpy.complex128)
    nonwritable.setflags(write=False)

    def __array__(_self, dtype=None):
        return numpy.asarray(nonwritable, dtype=dtype)

    def decorator(cls):
        if hasattr(cls, '__array__'):
            raise RuntimeError("Refusing to decorate a class that already has '__array__' defined.")
        cls.__array__ = __array__
        return cls
    return decorator