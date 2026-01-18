import numpy as np
from ase import Atoms
from ase.io import read, write
import ase.io.rmc6f as rmc6f
from ase.lattice.compounds import TRI_Fe2O3
def test_rmc6f_write_output_column_format():
    """Test for utility function that processes the columns in array
    and gets back out formatting information.
    """
    cols = ['id', 'symbols', 'scaled_positions', 'ref_num', 'ref_cell']
    arrays = {}
    arrays['id'] = np.array([1, 2, 3, 4, 5, 6, 7])
    arrays['symbols'] = np.array(symbols)
    arrays['ref_num'] = np.zeros(7, int)
    arrays['ref_cell'] = np.zeros((7, 3), int)
    arrays['scaled_positions'] = lat_positions / lat
    ncols, dtype_obj, fmt = rmc6f._write_output_column_format(cols, arrays)
    target_ncols = [1, 1, 3, 1, 3]
    target_fmt = '%8d %s%14.6f %14.6f %14.6f %8d %8d %8d %8d \n'
    assert ncols == target_ncols
    assert fmt == target_fmt