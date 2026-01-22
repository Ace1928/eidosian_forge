import warnings
from Bio import BiopythonDeprecationWarning
def _uniform_norm(shape):
    """Normalize a uniform matrix (PRIVATE)."""
    matrix = np.ones(shape)
    return _normalize(matrix)