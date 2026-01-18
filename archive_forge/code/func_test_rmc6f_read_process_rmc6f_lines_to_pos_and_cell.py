import numpy as np
from ase import Atoms
from ase.io import read, write
import ase.io.rmc6f as rmc6f
from ase.lattice.compounds import TRI_Fe2O3
def test_rmc6f_read_process_rmc6f_lines_to_pos_and_cell():
    """Test for utility function that processes lines of rmc6f using
    regular expressions to capture atom properties and cell information
    """
    tol = 1e-05
    lines = rmc6f_input_text.split('\n')
    props, cell = rmc6f._read_process_rmc6f_lines_to_pos_and_cell(lines)
    target_cell = np.zeros((3, 3), float)
    np.fill_diagonal(target_cell, 4.672816)
    assert props == symbol_xyz_dict
    assert np.allclose(cell, target_cell, rtol=tol)