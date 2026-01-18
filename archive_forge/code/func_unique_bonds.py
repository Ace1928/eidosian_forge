from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
@property
def unique_bonds(self):
    """Get Unique Bonds.

        :data:`all_bonds` i-j without j-i. This is the upper triangle of the
        connectivity matrix (i,j), `i < j`

        """
    bonds = []
    for imI in range(len(self.all_bonds)):
        bonds.append([])
        for iAtom, bonded in enumerate(self.all_bonds[imI]):
            bonds[-1].append([jAtom for jAtom in bonded if jAtom > iAtom])
    return bonds