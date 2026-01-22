import warnings
from Bio import BiopythonDeprecationWarning
class NaiveBayes:
    """Hold information for a NaiveBayes classifier.

    Attributes:
     - classes        - List of the possible classes of data.
     - p_conditional  - CLASS x DIM array of dicts of value -> ``P(value|class,dim)``
     - p_prior        - List of the prior probabilities for every class.
     - dimensionality - Dimensionality of the data.

    """

    def __init__(self):
        """Initialize the class."""
        self.classes = []
        self.p_conditional = None
        self.p_prior = []
        self.dimensionality = None