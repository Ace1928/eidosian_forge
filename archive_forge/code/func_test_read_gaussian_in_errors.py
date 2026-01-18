import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
@pytest.mark.parametrize('unsupported_file', [fd_incorrect_zmatrix_symbol(), fd_unsupported_option(), fd_no_charge_mult()])
def test_read_gaussian_in_errors(fd_command_set, unsupported_file):
    with pytest.raises(ParseError):
        read_gaussian_in(unsupported_file, True)