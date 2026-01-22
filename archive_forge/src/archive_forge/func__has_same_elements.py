from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _has_same_elements(self):
    """Check if two structures have same elements."""
    elem1 = self._get_element_count(self.s1)
    return elem1 == self._get_element_count(self.s2)