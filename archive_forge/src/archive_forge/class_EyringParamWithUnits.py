import math
from .._util import get_backend
from ..util.pyutil import defaultnamedtuple
from ..units import default_units, Backend, default_constants, format_string
from .arrhenius import _get_R, _fit
class EyringParamWithUnits(EyringParam):

    def __call__(self, state, constants=default_constants, units=default_units, backend=None):
        """See :func:`chempy.eyring.eyring_equation`."""
        if backend is None:
            backend = Backend()
        return super(EyringParamWithUnits, self).__call__(state, constants, units, backend)

    def as_RateExpr(self, unique_keys=None, constants=default_constants, units=default_units, backend=None):
        if backend is None:
            backend = Backend()
        return super(EyringParamWithUnits, self).as_RateExpr(unique_keys, constants, units, backend)