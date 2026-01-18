import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..utils import marginal_counts
from ..counts import Counts
def z_diagonal(dim, dtype=float):
    """Return the diagonal for the operator :math:`Z^\\otimes n`"""
    parity = np.zeros(dim, dtype=dtype)
    for i in range(dim):
        parity[i] = bin(i)[2:].count('1')
    return (-1) ** np.mod(parity, 2)