import numpy as np
import scipy.stats
import cirq
def _random_special_unitary():
    U = _random_unitary()
    return U / np.sqrt(np.linalg.det(U))