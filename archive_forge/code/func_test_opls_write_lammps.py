import numpy as np
import pytest
from ase.io.opls import OPLSff, OPLSStructure
def test_opls_write_lammps(opls_structure_file_name, opls_force_field_file_name):
    LAMMPS_FILES_PREFIX = 'lmp'
    atoms = OPLSStructure(opls_structure_file_name)
    with open(opls_force_field_file_name) as fd:
        opls_force_field = OPLSff(fd)
    opls_force_field.write_lammps(atoms, prefix=LAMMPS_FILES_PREFIX)
    with open(LAMMPS_FILES_PREFIX + '_atoms') as fd:
        lammps_data = fd.readlines()
    for ind, line in enumerate(lammps_data):
        if line.startswith('Atoms'):
            atom1_data = lammps_data[ind + 2]
            atom2_data = lammps_data[ind + 3]
            atom3_data = lammps_data[ind + 4]
            break
    pos_indices = slice(4, 7)
    atom1_pos = np.array(atom1_data.split()[pos_indices], dtype=float)
    atom2_pos = np.array(atom2_data.split()[pos_indices], dtype=float)
    atom3_pos = np.array(atom3_data.split()[pos_indices], dtype=float)
    assert atom1_pos == pytest.approx(np.array([1.6139, -0.7621, 0.0]), abs=0.0001)
    assert atom2_pos == pytest.approx(np.array([-0.3279, 0.5227, 0]), abs=0.0001)
    assert atom3_pos == pytest.approx(np.array([-0.96, 0.5809, 0.8875]), abs=0.0001)