import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from locale import getpreferredencoding
import numpy as np
from .affines import apply_affine, dot_reduce, from_matvec
from .eulerangles import euler2mat
from .fileslice import fileslice, strided_scalar
from .nifti1 import unit_codes
from .openers import ImageOpener
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import Recoder, array_from_file
class PARRECArrayProxy:

    def __init__(self, file_like, header, *, mmap=True, scaling='dv'):
        """Initialize PARREC array proxy

        Parameters
        ----------
        file_like : file-like object
            Filename or object implementing ``read, seek, tell``
        header : PARRECHeader instance
            Implementing ``get_data_shape, get_data_dtype``,
            ``get_sorted_slice_indices``, ``get_data_scaling``,
            ``get_rec_shape``.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading data.
            If False, do not try numpy ``memmap`` for data array.  If one of
            {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A `mmap` value of
            True gives the same behavior as ``mmap='c'``.  If `file_like`
            cannot be memory-mapped, ignore `mmap` value and read array from
            file.
        scaling : {'fp', 'dv'}, optional, keyword only
            Type of scaling to use - see header ``get_data_scaling`` method.
        """
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        self.file_like = file_like
        self._shape = header.get_data_shape()
        self._dtype = header.get_data_dtype()
        self._slice_indices = header.get_sorted_slice_indices()
        self._mmap = mmap
        self._slice_scaling = header.get_data_scaling(scaling)
        self._rec_shape = header.get_rec_shape()

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_proxy(self):
        return True

    def _get_unscaled(self, slicer):
        indices = self._slice_indices
        if slicer == ():
            with ImageOpener(self.file_like) as fileobj:
                rec_data = array_from_file(self._rec_shape, self._dtype, fileobj, mmap=self._mmap)
                rec_data = rec_data[..., indices]
                return rec_data.reshape(self._shape, order='F')
        elif indices[0] != 0 or np.any(np.diff(indices) != 1):
            return self._get_unscaled(())[slicer]
        with ImageOpener(self.file_like) as fileobj:
            return fileslice(fileobj, slicer, self._shape, self._dtype, 0, 'F')

    def _get_scaled(self, dtype, slicer):
        raw_data = self._get_unscaled(slicer)
        if self._slice_scaling is None:
            if dtype is None:
                return raw_data
            final_type = np.promote_types(raw_data.dtype, dtype)
            return raw_data.astype(final_type, copy=False)
        fake_data = strided_scalar(self._shape)
        _, slopes, inters = np.broadcast_arrays(fake_data, *self._slice_scaling)
        final_type = np.result_type(raw_data, slopes, inters)
        if dtype is not None:
            final_type = np.promote_types(final_type, dtype)
        return raw_data * slopes[slicer].astype(final_type) + inters[slicer].astype(final_type)

    def get_unscaled(self):
        """Read data from file

        This is an optional part of the proxy API
        """
        return self._get_unscaled(slicer=())

    def __array__(self, dtype=None):
        """Read data from file and apply scaling, casting to ``dtype``

        If ``dtype`` is unspecified, the dtype of the returned array is the
        narrowest dtype that can represent the data without overflow.
        Generally, it is the wider of the dtypes of the slopes or intercepts.

        Parameters
        ----------
        dtype : numpy dtype specifier, optional
            A numpy dtype specifier specifying the type of the returned array.

        Returns
        -------
        array
            Scaled image data with type `dtype`.
        """
        arr = self._get_scaled(dtype=dtype, slicer=())
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, slicer):
        return self._get_scaled(dtype=None, slicer=slicer)