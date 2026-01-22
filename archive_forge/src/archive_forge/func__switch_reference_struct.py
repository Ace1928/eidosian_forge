from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _switch_reference_struct(self):
    """There is an intrinsic assymetry in the system because
        one of the atoms are being expanded, while the other is not.
        This can cause the algorithm to return different result
        depending on which structure is passed first.
        We adopt the convention of using the atoms object
        having the fewest atoms in its expanded cell as the
        reference object.
        We return True if a switch of structures has been performed."""
    if self.expanded_s1 is None:
        self.expanded_s1 = self._expand(self.s1)
    if self.expanded_s2 is None:
        self.expanded_s2 = self._expand(self.s2)
    exp1 = self.expanded_s1
    exp2 = self.expanded_s2
    if len(exp1) < len(exp2):
        s1_temp = self.s1.copy()
        self.s1 = self.s2
        self.s2 = s1_temp
        exp1_temp = self.expanded_s1.copy()
        self.expanded_s1 = self.expanded_s2
        self.expanded_s2 = exp1_temp
        return True
    return False