from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from .. import load as top_load
from ..imageclasses import spatial_axes_first
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..testing import bytesio_round_trip, deprecated_to, expires, memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
class CHeader(SpatialHeader):
    data_layout = 'C'