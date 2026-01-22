from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
def _fit_linearized(backtransfm, lin_x, lin_y, lin_yerr):
    if len(lin_x) != len(lin_y):
        raise ValueError('k and T needs to be of equal length.')
    if lin_yerr is not None:
        if len(lin_yerr) != len(lin_y):
            raise ValueError('kerr and T needs to be of equal length.')
    lin_p, lin_vcv, lin_r2 = least_squares(lin_x, lin_y, lin_yerr)
    return [cb(lin_p) for cb in backtransfm]