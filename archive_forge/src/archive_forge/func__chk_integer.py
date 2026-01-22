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
@staticmethod
def _chk_integer(hdr, fix=False):
    rep = Report(HeaderDataError)
    if hdr['an_integer'] == 1:
        return (hdr, rep)
    rep.problem_level = 40
    rep.problem_msg = 'an_integer should be 1'
    if fix:
        hdr['an_integer'] = 1
        rep.fix_msg = 'set an_integer to 1'
    return (hdr, rep)