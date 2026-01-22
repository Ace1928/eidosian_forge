import warnings
from Bio import BiopythonDeprecationWarning
def _exp_logsum(numbers):
    """Return the exponential of a logsum (PRIVATE)."""
    sum = _logsum(numbers)
    return np.exp(sum)