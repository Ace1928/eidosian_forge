import numpy as np
from ase import Atoms
from ase.io import read, write
import ase.io.rmc6f as rmc6f
from ase.lattice.compounds import TRI_Fe2O3
def test_rmc6f_read():
    """Test for reading rmc6f input file."""
    with open('input.rmc6f', 'w') as rmc6f_input_f:
        rmc6f_input_f.write(rmc6f_input_text)
    rmc6f_input_atoms = read('input.rmc6f')
    assert len(rmc6f_input_atoms) == 7
    assert rmc6f_input_atoms == rmc6f_atoms