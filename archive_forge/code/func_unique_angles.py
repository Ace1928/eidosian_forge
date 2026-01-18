from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
@property
def unique_angles(self):
    """Get Unique Angles.

        :data:`all_angles` i-j-k without k-j-i.

        """
    return self._filter_unique(self.all_angles)