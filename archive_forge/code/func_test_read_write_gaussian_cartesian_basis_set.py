import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def test_read_write_gaussian_cartesian_basis_set(fd_cartesian_basis_set, cartesian_setup):
    atoms, params = cartesian_setup
    atoms.pbc = None
    atoms.cell = None
    iso_params = {'temperature': '300', 'pressure': '1.0', 'scale': '1.0'}
    params.update(iso_params)
    params['opt'] = 'tight maxcyc=100'
    params['frequency'] = 'anharmonic'
    params['basis'] = 'gen'
    params['method'] = 'g1'
    params['fitting_basis'] = 'tzvpfit'
    params['save'] = ''
    params['basis_set'] = _basis_set_text
    atoms_new = read_gaussian_in(fd_cartesian_basis_set, True)
    atoms_new.set_masses(_get_iso_masses(atoms_new))
    _check_atom_properties(atoms, atoms_new, params)
    with pytest.warns(UserWarning):
        _test_write_gaussian(atoms_new, params)