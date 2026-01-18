import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def test_read_write_gaussian_zmatrix(fd_zmatrix):
    positions = np.array([[+0.0, +0.0, +0.0], [+1.31, +0.0, +0.0], [-0.16, +1.3, +0.0], [+1.15, +1.3, +0.0], [-0.394, -0.446, +1.031], [-0.394, -0.446, -1.031], [+1.545, +1.746, -1.031], [+1.545, +1.746, +1.031]])
    masses = [None] * 8
    masses[1] = 0.1134289259
    atoms = Atoms('BH2BH4', positions=positions, masses=masses)
    params = {'chk': 'example.chk', 'nprocshared': '16', 'output_type': 't', 'b3lyp': None, 'gen': None, 'opt': 'tight, maxcyc=100', 'freq': None, 'integral': 'ultrafine', 'charge': 0, 'mult': 1, 'temperature': '300', 'pressure': '1.0', 'basisfile': '@basis-set-filename.gbs'}
    params['isolist'] = np.array(masses)
    atoms_new = read_gaussian_in(fd_zmatrix, True)
    atoms_new.set_masses(_get_iso_masses(atoms_new))
    _check_atom_properties(atoms, atoms_new, params)
    params['output_type'] = 'p'
    _test_write_gaussian(atoms_new, params)