import functools
import itertools
from collections.abc import Iterable, Sequence
import numpy as np
from pennylane.pytrees import register_pytree
def toset(self):
    """Returns a set representation of the Wires object.

        Returns:
            Set: set of wire labels
        """
    return set(self.labels)