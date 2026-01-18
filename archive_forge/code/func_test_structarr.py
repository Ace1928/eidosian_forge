import logging
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import imageglobals
from ..batteryrunners import Report
from ..casting import sctypes
from ..spatialimages import HeaderDataError
from ..volumeutils import Recoder, native_code, swapped_code
from ..wrapstruct import LabeledWrapStruct, WrapStruct, WrapStructError
def test_structarr(self):
    hdr = self.header_class()
    hdr.structarr
    with pytest.raises(AttributeError):
        hdr.structarr = 0