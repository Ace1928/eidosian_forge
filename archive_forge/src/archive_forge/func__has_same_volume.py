from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _has_same_volume(self):
    vol1 = self.s1.get_volume()
    vol2 = self.s2.get_volume()
    return np.abs(vol1 - vol2) < self.vol_tol