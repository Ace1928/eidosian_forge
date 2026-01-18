import numpy as np
from qiskit.exceptions import QiskitError
Format unitary coming from the backend to present to the Qiskit user.

    Args:
        mat (list[list]): a list of list of [re, im] complex numbers
        decimals (int): the number of decimals in the statevector.
            If None, no rounding is done.

    Returns:
        list[list[complex]]: a matrix of complex numbers
    