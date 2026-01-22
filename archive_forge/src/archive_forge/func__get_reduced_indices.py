from typing import List
import numpy as np
from ase import Atoms
from .spacegroup import Spacegroup, _SPACEGROUP
def _get_reduced_indices(atoms: Atoms, tol: float=1e-05) -> List[int]:
    """Get a list of the reduced atomic indices using spglib.
    Note: Does no checks to see if spglib is installed.
    
    :param atoms: ase Atoms object to reduce
    :param tol: ``float``, numeric tolerance for positional comparisons
    """
    import spglib
    spglib_cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.numbers)
    symmetry_data = spglib.get_symmetry_dataset(spglib_cell, symprec=tol)
    return list(set(symmetry_data['equivalent_atoms']))