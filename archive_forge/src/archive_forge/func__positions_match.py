from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _positions_match(self, rotation_reflection_matrices, translations):
    """Check if the position and elements match.

        Note that this function changes self.s1 and self.s2 to the rotation and
        translation that matches best. Hence, it is crucial that this function
        calls the element comparison, not the other way around.
        """
    pos1_ref = self.s1.get_positions(wrap=True)
    exp2 = self.expanded_s2
    tree = KDTree(exp2.get_positions())
    for i in range(translations.shape[0]):
        pos1_trans = pos1_ref - translations[i]
        for matrix in rotation_reflection_matrices:
            pos1 = matrix.dot(pos1_trans.T).T
            self.s1.set_positions(pos1)
            self.s1.wrap(pbc=[1, 1, 1])
            if self._elements_match(self.s1, exp2, tree):
                return True
    return False