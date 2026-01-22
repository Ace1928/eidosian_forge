import warnings
import numpy as np
from .minc1 import Minc1File, Minc1Image, MincError, MincHeader
class Minc2File(Minc1File):
    """Class to wrap MINC2 format file

    Although it has some of the same methods as a ``Header``, we use
    this only when reading a MINC2 file, to pull out useful header
    information, and for the method of reading the data out
    """

    def __init__(self, mincfile):
        self._mincfile = mincfile
        minc_part = mincfile['minc-2.0']
        image = minc_part['image']['0']
        self._image = image['image']
        self._dim_names = self._get_dimensions(self._image)
        dimensions = minc_part['dimensions']
        self._dims = [Hdf5Bunch(dimensions[s]) for s in self._dim_names]
        for dim in self._dims:
            spacing = getattr(dim, 'spacing', b'regular__')
            if spacing == b'irregular':
                raise ValueError('Irregular spacing not supported')
            elif spacing != b'regular__':
                warnings.warn(f'Invalid spacing declaration: {spacing}; assuming regular')
        self._spatial_dims = [name for name in self._dim_names if name.endswith('space')]
        self._image_max = image['image-max']
        self._image_min = image['image-min']

    def _get_dimensions(self, var):
        try:
            dimorder = var.attrs['dimorder'].decode()
        except KeyError:
            return []
        return dimorder.split(',')[:len(var.shape)]

    def get_data_dtype(self):
        return self._image.dtype

    def get_data_shape(self):
        return self._image.shape

    def _get_valid_range(self):
        """Return valid range for image data

        The valid range can come from the image 'valid_range' or
        failing that, from the data type range
        """
        ddt = self.get_data_dtype()
        info = np.iinfo(ddt.type)
        try:
            valid_range = self._image.attrs['valid_range']
        except (AttributeError, KeyError):
            valid_range = [info.min, info.max]
        else:
            if valid_range[0] < info.min or valid_range[1] > info.max:
                raise ValueError('Valid range outside input data type range')
        return np.asarray(valid_range, dtype=np.float64)

    def _get_scalar(self, var):
        """Get scalar value from HDF5 scalar"""
        return var[()]

    def _get_array(self, var):
        """Get array from HDF5 array"""
        return np.asanyarray(var)

    def get_scaled_data(self, sliceobj=()):
        """Return scaled data for slice definition `sliceobj`

        Parameters
        ----------
        sliceobj : tuple, optional
            slice definition. If not specified, return whole array

        Returns
        -------
        scaled_arr : array
            array from minc file with scaling applied
        """
        if sliceobj == ():
            raw_data = np.asanyarray(self._image)
        else:
            try:
                raw_data = self._image[sliceobj]
            except (ValueError, TypeError):
                raw_data = np.asanyarray(self._image)[sliceobj]
            else:
                raw_data = np.asanyarray(raw_data)
        return self._normalize(raw_data, sliceobj)