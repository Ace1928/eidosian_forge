from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
@property
def all_bonds(self):
    """All Bonds.

        A list with indices of bonded atoms for each neighborlist in *self*.
        Atom i is connected to all atoms inside result[i]. Duplicates from PBCs are
        removed. See also :data:`unique_bonds`.

        **No setter or deleter, only getter**
        """
    if not 'allBonds' in self._cache:
        self._cache['allBonds'] = self._get_all_x(1)
    return self._cache['allBonds']