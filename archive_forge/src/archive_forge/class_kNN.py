import warnings
from Bio import BiopythonDeprecationWarning
class kNN:
    """Holds information necessary to do nearest neighbors classification.

    Attributes:
     - classes  Set of the possible classes.
     - xs       List of the neighbors.
     - ys       List of the classes that the neighbors belong to.
     - k        Number of neighbors to look at.
    """

    def __init__(self):
        """Initialize the class."""
        self.classes = set()
        self.xs = []
        self.ys = []
        self.k = None