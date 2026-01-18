import sys
import h5py
import numpy as np
from . import core
Creates a new variable.

        Parameters
        ----------
        varname : str
            Name of the new variable. If given as a path, intermediate groups will be created,
            if not existent.
        datatype : numpy.dtype, str
            Dataype of the new variable
        dimensions : tuple
            Tuple containing dimension name strings. Defaults to empty tuple, effectively
            creating a scalar variable.
        zlib : bool, optional
            If ``True``, variable data will be gzip compressed.
        complevel : int, optional
            Integer between 1 and 9 defining compression level. Defaults to 4.
            Ignored if ``zlib=False``.
        shuffle : bool, optional
            If ``True``, HDF5 shuffle filter will be applied. Defaults to ``True``.
            Ignored if ``zlib=False``.
        fletcher32 : bool, optional
            If ``True``, HDF5 Fletcher32 checksum algorithm is applied. Defaults to ``False``.
        chunksizes : tuple, optional
            Tuple of integers specifying the chunksizes of each variable dimension.
            Discussion on ``h5netcdf`` chunksizes can be found in (:issue:`52`) and (:pull:`127`).
        fill_value : scalar, optional
            Specify ``_FillValue`` for uninitialized parts of the variable. Defaults to ``None``.
        endian : str, optional
            Control on-disk storage format.
            Can be any of ``little``, ``big`` or ``native`` (default).

        Returns
        -------
        var : h5netcdf.legacyapi.Variable
            Variable class instance
        