import numpy as np
from .analyze import AnalyzeHeader
from .batteryrunners import Report
from .filebasedimages import ImageFileError
from .nifti1 import Nifti1Header, Nifti1Image, Nifti1Pair
from .spatialimages import HeaderDataError
@staticmethod
def _chk_eol_check(hdr, fix=False):
    rep = Report(HeaderDataError)
    if np.all(hdr['eol_check'] == (13, 10, 26, 10)):
        return (hdr, rep)
    if np.all(hdr['eol_check'] == 0):
        rep.problem_level = 20
        rep.problem_msg = 'EOL check all 0'
        if fix:
            hdr['eol_check'] = (13, 10, 26, 10)
            rep.fix_msg = 'setting EOL check to 13, 10, 26, 10'
        return (hdr, rep)
    rep.problem_level = 40
    rep.problem_msg = 'EOL check not 0 or 13, 10, 26, 10; data may be corrupted by EOL conversion'
    if fix:
        hdr['eol_check'] = (13, 10, 26, 10)
        rep.fix_msg = 'setting EOL check to 13, 10, 26, 10'
    return (hdr, rep)