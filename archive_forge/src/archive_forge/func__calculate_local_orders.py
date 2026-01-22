import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def _calculate_local_orders(self, individual_fingerprints, typedic, volume):
    """ Returns a list with the local order for every atom,
        using the definition of local order from
        Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632
        https://doi.org/10.1016/j.cpc.2010.06.007"""
    n_tot = sum([len(typedic[key]) for key in typedic])
    local_orders = []
    for index, fingerprints in individual_fingerprints.items():
        local_order = 0
        for unique_type, fingerprint in fingerprints.items():
            term = np.linalg.norm(fingerprint) ** 2
            term *= self.binwidth
            term *= (volume * 1.0 / n_tot) ** 3
            term *= len(typedic[unique_type]) * 1.0 / n_tot
            local_order += term
        local_orders.append(np.sqrt(local_order))
    return local_orders