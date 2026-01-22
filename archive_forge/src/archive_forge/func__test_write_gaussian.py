import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def _test_write_gaussian(atoms, params_expected, properties=None):
    """Writes atoms to gaussian input file, reads this back in and
    checks that the resulting atoms object is equal to atoms and
    the calculator has parameters params_expected"""
    atoms.calc.label = 'gaussian_input_file'
    out_file = atoms.calc.label + '.com'
    atoms_expected = atoms.copy()
    atoms.calc.write_input(atoms, properties=properties)
    with open(out_file) as fd:
        atoms_written = read_gaussian_in(fd, True)
        if _get_iso_masses(atoms_written):
            atoms_written.set_masses(_get_iso_masses(atoms_written))
    _check_atom_properties(atoms_expected, atoms_written, params_expected)