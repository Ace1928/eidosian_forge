import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def _json_decode(self, fingerprints, typedic):
    """ This is the reverse operation of _json_encode """
    fingerprints_decoded = {}
    for key, val in fingerprints.items():
        newkey = list(map(int, key.split('_')))
        if len(newkey) > 1:
            newkey = tuple(newkey)
        else:
            newkey = newkey[0]
        if isinstance(val, dict):
            fingerprints_decoded[newkey] = {}
            for key2, val2 in val.items():
                fingerprints_decoded[newkey][int(key2)] = np.array(val2)
        else:
            fingerprints_decoded[newkey] = np.array(val)
    typedic_decoded = {}
    for key, val in typedic.items():
        newkey = int(key)
        typedic_decoded[newkey] = val
    return [fingerprints_decoded, typedic_decoded]