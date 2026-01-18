import contextlib
import pennylane as qml
from pennylane.measurements import Shots
Context manager to temporarily change the shots
    of a device.

    This context manager can be used in two ways.

    As a standard context manager:

    >>> dev = qml.device("default.qubit.legacy", wires=2, shots=None)
    >>> with set_shots(dev, shots=100):
    ...     print(dev.shots)
    100
    >>> print(dev.shots)
    None

    Or as a decorator that acts on a function that uses the device:

    >>> set_shots(dev, shots=100)(lambda: dev.shots)()
    100
    