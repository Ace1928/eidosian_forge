import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, NaN
from .umath import log10, exp2, nextafter, isnan
def _float_to_str(self, value):
    """Converts float to str.

        Parameters
        ----------
        value : float
            value to be converted.
        """
    return self.params['fmt'] % array(_fr0(value)[0], self.ftype)