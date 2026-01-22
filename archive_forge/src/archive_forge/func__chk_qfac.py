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
def _chk_qfac(hdr, fix=False):
    rep = Report(HeaderDataError)
    if hdr['pixdim'][0] in (-1, 0, 1):
        return (hdr, rep)
    rep.problem_level = 20
    rep.problem_msg = 'pixdim[0] (qfac) should be 1 (default) or 0 or -1'
    if fix:
        hdr['pixdim'][0] = 1
        rep.fix_msg = 'setting qfac to 1'
    return (hdr, rep)