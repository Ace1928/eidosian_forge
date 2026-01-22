import warnings
from Bio import BiopythonDeprecationWarning
def _argmaxes(vector, allowance=None):
    """Return indices of the maximum values aong the vector (PRIVATE)."""
    return [np.argmax(vector)]