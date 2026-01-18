import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
def to_fileobj(self, fileobj, order='F'):
    """Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        """
    mn, mx = self._writing_range()
    array_to_file(self._array, fileobj, self._out_dtype, offset=None, intercept=self.inter, divslope=self.slope, mn=mn, mx=mx, order=order, nan2zero=self._needs_nan2zero())