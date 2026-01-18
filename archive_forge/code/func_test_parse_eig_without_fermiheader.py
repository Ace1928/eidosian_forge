from io import StringIO
import numpy as np
import pytest
from ase.io import read, write
from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.abinit import read_abinit_out, read_eig, match_kpt_header
from ase.units import Hartree, Bohr
def test_parse_eig_without_fermiheader():
    fd = StringIO(eig_text)
    next(fd)
    data = read_eig(fd)
    assert 'fermilevel' not in data
    assert {'eigenvalues', 'ibz_kpoints', 'kpoint_weights'} == set(data)