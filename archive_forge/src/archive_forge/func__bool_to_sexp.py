from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _bool_to_sexp(val: bool):
    rlib = openrlib.rlib
    s = rlib.Rf_protect(rlib.Rf_allocVector(rlib.LGLSXP, 1))
    openrlib.SET_LOGICAL_ELT(s, 0, int(val))
    rlib.Rf_unprotect(1)
    return s