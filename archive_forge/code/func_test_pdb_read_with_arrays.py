import numpy as np
import pytest
from ase import io
def test_pdb_read_with_arrays():
    """Read information from pdb file. Includes occupancy."""
    with open('pdb_test_2.pdb', 'w') as pdb_file:
        pdb_file.write('\n'.join(test_pdb.splitlines()[:6]))
    expected_occupancy = [0.0, 0.0, 1.0, 0.4]
    expected_bfactor = [0.0, 0.0, 0.0, 38.51]
    pdb_atoms = io.read('pdb_test_2.pdb')
    assert len(pdb_atoms) == 4
    assert np.allclose(pdb_atoms.arrays['occupancy'], expected_occupancy)
    assert np.allclose(pdb_atoms.arrays['bfactor'], expected_bfactor)