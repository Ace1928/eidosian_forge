import warnings
from io import BytesIO
import numpy as np
from . import analyze  # module import
from .batteryrunners import Report
from .optpkg import optional_package
from .spatialimages import HeaderDataError, HeaderTypeError
@staticmethod
def _chk_origin(hdr, fix=False):
    rep = Report(HeaderDataError)
    origin = hdr['origin'][0:3]
    dims = hdr['dim'][1:4]
    if not np.any(origin) or (np.all(origin > -dims) and np.all(origin < dims * 2)):
        return (hdr, rep)
    rep.problem_level = 20
    rep.problem_msg = 'very large origin values relative to dims'
    if fix:
        rep.fix_msg = 'leaving as set, ignoring for affine'
    return (hdr, rep)