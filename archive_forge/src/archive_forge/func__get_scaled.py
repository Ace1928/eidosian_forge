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