import math
from .._util import get_backend
from ..util.pyutil import defaultnamedtuple
from ..units import default_units, Backend, default_constants, format_string
from .arrhenius import _get_R, _fit
def _get_kB_over_h(constants=None, units=None):
    if constants is None:
        kB_over_h = 20836643994.118652
        if units is not None:
            s = units.second
            K = units.kelvin
            kB_over_h /= s * K
    else:
        kB_over_h = constants.Boltzmann_constant / constants.Planck_constant
    return kB_over_h