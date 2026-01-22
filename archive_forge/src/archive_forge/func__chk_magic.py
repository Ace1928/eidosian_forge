from __future__ import annotations
import warnings
from io import BytesIO
import numpy as np
import numpy.linalg as npl
from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes
@staticmethod
def _chk_magic(hdr, fix=False):
    rep = Report(HeaderDataError)
    magic = hdr['magic'].item()
    if magic in (hdr.pair_magic, hdr.single_magic):
        return (hdr, rep)
    rep.problem_msg = f'magic string {magic.decode('latin1')!r} is not valid'
    rep.problem_level = 45
    if fix:
        rep.fix_msg = 'leaving as is, but future errors are likely'
    return (hdr, rep)