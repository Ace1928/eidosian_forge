from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
def _filter_unique(self, l):
    """Helper function to filter for unique lists in a list
        that also contains the reversed items.
        """
    r = []
    for imI in range(len(l)):
        r.append([])
        for i, tuples in enumerate(l[imI]):
            r[-1].append([x for x in tuples if i < x[-1]])
    return r