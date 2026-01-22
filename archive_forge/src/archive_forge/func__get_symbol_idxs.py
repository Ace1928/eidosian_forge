from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
def _get_symbol_idxs(self, imI, sym):
    """Get list of indices of element *sym*"""
    if isinstance(imI, int):
        return [idx for idx in range(len(self.images[imI])) if self.images[imI][idx].symbol == sym]
    else:
        return [idx for idx in range(len(imI)) if imI[idx].symbol == sym]