import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
class affine_penalty:
    """Create a gap function for use in an alignment."""

    def __init__(self, open, extend, penalize_extend_when_opening=0):
        """Initialize the class."""
        if open > 0 or extend > 0:
            raise ValueError('Gap penalties should be non-positive.')
        if not penalize_extend_when_opening and extend < open:
            raise ValueError('Gap opening penalty should be higher than gap extension penalty (or equal)')
        self.open, self.extend = (open, extend)
        self.penalize_extend_when_opening = penalize_extend_when_opening

    def __call__(self, index, length):
        """Call a gap function instance already created."""
        return calc_affine_penalty(length, self.open, self.extend, self.penalize_extend_when_opening)