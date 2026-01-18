import numpy as np
from ase import Atoms
from ase.io import read, write
import ase.io.rmc6f as rmc6f
from ase.lattice.compounds import TRI_Fe2O3
def test_rmc6f_read_construct_regex():
    """Test for utility function that constructs rmc6f header regex."""
    header_lines = ['Number of atoms:', '  Supercell dimensions:  ', '    Cell (Ang/deg):  ', '      Lattice vectors (Ang):  ']
    result = rmc6f._read_construct_regex(header_lines)
    target = '(Number\\s+of\\s+atoms:|Supercell\\s+dimensions:|Cell\\s+\\(Ang/deg\\):|Lattice\\s+vectors\\s+\\(Ang\\):)'
    assert result == target