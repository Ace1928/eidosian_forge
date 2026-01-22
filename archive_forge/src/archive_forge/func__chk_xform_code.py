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
@classmethod
def _chk_xform_code(klass, code_type, hdr, fix):
    rep = Report(HeaderDataError)
    code = int(hdr[code_type])
    recoder = klass._field_recoders[code_type]
    if code in recoder.value_set():
        return (hdr, rep)
    rep.problem_level = 30
    rep.problem_msg = '%s %d not valid' % (code_type, code)
    if fix:
        hdr[code_type] = 0
        rep.fix_msg = 'setting to 0'
    return (hdr, rep)