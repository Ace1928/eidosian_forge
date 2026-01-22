import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
def _get_oriented_data(self, raw_data, orientation=None):
    """
        Get data oriented following ``patient_orientation`` header field. If
        the ``orientation`` parameter is given, return data according to this
        orientation.

        :param raw_data: Numpy array containing the raw data
        :param orientation: None (default), 'neurological' or 'radiological'
        :rtype: Numpy array containing the oriented data
        """
    if orientation is None:
        orientation = self._header['patient_orientation']
    elif orientation == 'neurological':
        orientation = patient_orient_neurological[0]
    elif orientation == 'radiological':
        orientation = patient_orient_radiological[0]
    else:
        raise ValueError('orientation should be None, neurological or radiological')
    if orientation in patient_orient_neurological:
        raw_data = raw_data[::-1, ::-1, ::-1]
    elif orientation in patient_orient_radiological:
        raw_data = raw_data[:, ::-1, ::-1]
    return raw_data