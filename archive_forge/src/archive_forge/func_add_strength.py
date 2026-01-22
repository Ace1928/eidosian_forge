from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from qiskit.exceptions import QiskitError
from .utilities import EPSILON
def add_strength(self, new_strength: float=0.0):
    """
        Returns a new XXPolytope with one new XX interaction appended.
        """
    return XXPolytope(total_strength=self.total_strength + new_strength, max_strength=max(self.max_strength, new_strength), place_strength=self.max_strength if new_strength > self.max_strength else new_strength if new_strength > self.place_strength else self.place_strength)