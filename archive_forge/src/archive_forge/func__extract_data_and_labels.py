from itertools import product
import numpy as np
from .utils import format_nvec
def _extract_data_and_labels(coeffs):
    """Helper function for creating frequency labels and partitioning data.

    Args:
        coeffs (array[complex]): A list of sets of Fourier coefficients.

    Returns:
        (list(str), dict[str, array[complex]): The set of frequency labels, and a data
            dictionary split into real and imaginary parts.
    """
    nvecs = list(product(*(np.array(range(-(d // 2), d // 2 + 1)) for d in coeffs[0].shape)))
    nvecs_formatted = [format_nvec(nvec) for nvec in nvecs]
    data = {}
    data['real'] = np.array([[c[nvec] for nvec in nvecs] for c in coeffs.real])
    data['imag'] = np.array([[c[nvec] for nvec in nvecs] for c in coeffs.imag])
    return (nvecs_formatted, data)