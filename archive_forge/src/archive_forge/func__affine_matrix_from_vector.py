import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
def _affine_matrix_from_vector(v):
    """Affine matrix from linearized (d, d + 1) matrix entries."""
    nparam = v.size
    d = (1 + np.sqrt(1 + 4 * nparam)) / 2 - 1
    dimensionality = int(np.round(d))
    if d != dimensionality:
        raise ValueError(f'Invalid number of elements for linearized matrix: {nparam}')
    matrix = np.eye(dimensionality + 1)
    matrix[:-1, :] = np.reshape(v, (dimensionality, dimensionality + 1))
    return matrix