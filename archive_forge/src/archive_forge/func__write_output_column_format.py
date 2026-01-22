import re
import time
import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
from ase.cell import Cell
def _write_output_column_format(columns, arrays):
    """
    Helper function to build output for data columns in rmc6f format

    Parameters
    ----------
    columns: list[str]
        List of keys in arrays. Will be columns in the output file.
    arrays: dict{str:np.array}
        Dict with arrays for each column of rmc6f file that are
        property of Atoms object.

    Returns
    ------
    property_ncols : list[int]
        Number of columns for each property.
    dtype_obj: np.dtype
        Data type object for the columns.
    formats_as_str: str
        The format for printing the columns.

    """
    fmt_map = {'d': ('R', '%14.6f '), 'f': ('R', '%14.6f '), 'i': ('I', '%8d '), 'O': ('S', '%s'), 'S': ('S', '%s'), 'U': ('S', '%s'), 'b': ('L', ' %.1s ')}
    property_types = []
    property_ncols = []
    dtypes = []
    formats = []
    for column in columns:
        array = arrays[column]
        dtype = array.dtype
        property_type, fmt = fmt_map[dtype.kind]
        property_types.append(property_type)
        is_1d = len(array.shape) == 1
        is_1d_as_2d = len(array.shape) == 2 and array.shape[1] == 1
        if is_1d or is_1d_as_2d:
            ncol = 1
            dtypes.append((column, dtype))
        else:
            ncol = array.shape[1]
            for c in range(ncol):
                dtypes.append((column + str(c), dtype))
        formats.extend([fmt] * ncol)
        property_ncols.append(ncol)
    dtype_obj = np.dtype(dtypes)
    formats_as_str = ''.join(formats) + '\n'
    return (property_ncols, dtype_obj, formats_as_str)