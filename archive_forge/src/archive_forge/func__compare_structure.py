import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def _compare_structure(self, a1, a2):
    """ Returns the cosine distance between the two structures,
            using their fingerprints. """
    if len(a1) != len(a2):
        raise Exception('The two configurations are not the same size.')
    a1top = a1[-self.n_top:]
    a2top = a2[-self.n_top:]
    if 'fingerprints' in a1.info and (not self.recalculate):
        fp1, typedic1 = a1.info['fingerprints']
        fp1, typedic1 = self._json_decode(fp1, typedic1)
    else:
        fp1, typedic1 = self._take_fingerprints(a1top)
        a1.info['fingerprints'] = self._json_encode(fp1, typedic1)
    if 'fingerprints' in a2.info and (not self.recalculate):
        fp2, typedic2 = a2.info['fingerprints']
        fp2, typedic2 = self._json_decode(fp2, typedic2)
    else:
        fp2, typedic2 = self._take_fingerprints(a2top)
        a2.info['fingerprints'] = self._json_encode(fp2, typedic2)
    if sorted(fp1) != sorted(fp2):
        raise AssertionError('The two structures have fingerprints with different compounds.')
    for key in typedic1:
        if not np.array_equal(typedic1[key], typedic2[key]):
            raise AssertionError('The two structures have a different stoichiometry or ordering!')
    cos_dist = self._cosine_distance(fp1, fp2, typedic1)
    return cos_dist