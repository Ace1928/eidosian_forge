import warnings
import numpy as np
def write_record(self, *items):
    """
        Write a record (including sizes) to the file.

        Parameters
        ----------
        *items : array_like
            The data arrays to write.

        Notes
        -----
        Writes data items to a file::

            write_record(a.T, b.T, c.T, ...)

            write(1) a, b, c, ...

        Note that data in multidimensional arrays is written in
        row-major order --- to make them read correctly by Fortran
        programs, you need to transpose the arrays yourself when
        writing them.

        """
    items = tuple((np.asarray(item) for item in items))
    total_size = sum((item.nbytes for item in items))
    nb = np.array([total_size], dtype=self._header_dtype)
    nb.tofile(self._fp)
    for item in items:
        item.tofile(self._fp)
    nb.tofile(self._fp)