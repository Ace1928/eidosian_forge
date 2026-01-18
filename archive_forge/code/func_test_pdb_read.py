import numpy as np
import pytest
from ase import io
@pytest.mark.filterwarnings('ignore:Length of occupancy array')
def test_pdb_read():
    """Read information from pdb file."""
    with open('pdb_test.pdb', 'w') as pdb_file:
        pdb_file.write(test_pdb)
    expected_cell = [[30.0, 0.0, 0.0], [0.0, 15000.0, 0.0], [0.0, 0.0, 15000.0]]
    expected_positions = [[1.0, 8.0, 12.0], [2.0, 6.0, 4.0], [2.153, 14.096, 3.635], [3.846, 5.672, 1.323], [-2.481, 5.354, 0.0], [-11.713, -201.677, 9.06]]
    expected_species = ['C', 'C', 'Si', 'O', 'C', 'Si']
    pdb_atoms = io.read('pdb_test.pdb')
    assert len(pdb_atoms) == 6
    assert np.allclose(pdb_atoms.cell, expected_cell)
    assert np.allclose(pdb_atoms.positions, expected_positions)
    assert pdb_atoms.get_chemical_symbols() == expected_species
    assert 'occupancy' not in pdb_atoms.arrays