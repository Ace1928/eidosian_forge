import warnings
from Bio import BiopythonDeprecationWarning
class MarkovModel:
    """Create a state-emitting MarkovModel object."""

    def __init__(self, states, alphabet, p_initial=None, p_transition=None, p_emission=None):
        """Initialize the class."""
        self.states = states
        self.alphabet = alphabet
        self.p_initial = p_initial
        self.p_transition = p_transition
        self.p_emission = p_emission

    def __str__(self):
        """Create a string representation of the MarkovModel object."""
        from io import StringIO
        handle = StringIO()
        save(self, handle)
        handle.seek(0)
        return handle.read()