from io import BytesIO
import numpy as np
from packaging.version import Version, parse
from .. import xmlutils as xml
from ..batteryrunners import Report
from ..nifti1 import Nifti1Extension, extension_codes, intent_codes
from ..nifti2 import Nifti2Header, Nifti2Image
from ..spatialimages import HeaderDataError
from .cifti2 import (
@staticmethod
def _chk_pixdims(hdr, fix=False):
    rep = Report(HeaderDataError)
    pixdims = hdr['pixdim']
    spat_dims = pixdims[1:4]
    if not np.any(spat_dims < 0):
        return (hdr, rep)
    rep.problem_level = 35
    rep.problem_msg = 'pixdim[1,2,3] should be zero or positive'
    if fix:
        hdr['pixdim'][1:4] = np.abs(spat_dims)
        rep.fix_msg = 'setting to abs of pixdim values'
    return (hdr, rep)