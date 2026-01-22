import numpy as np
from qiskit.exceptions import QiskitError
def _list_to_complex_array(complex_list):
    """Convert nested list of shape (..., 2) to complex numpy array with shape (...)

    Args:
        complex_list (list): List to convert.

    Returns:
        np.ndarray: Complex numpy array

    Raises:
        QiskitError: If inner most array of input nested list is not of length 2.
    """
    arr = np.asarray(complex_list, dtype=np.complex128)
    if not arr.shape[-1] == 2:
        raise QiskitError('Inner most nested list is not of length 2.')
    return arr[..., 0] + 1j * arr[..., 1]