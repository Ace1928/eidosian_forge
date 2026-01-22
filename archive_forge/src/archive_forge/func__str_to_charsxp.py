from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _str_to_charsxp(val: Optional[str]):
    rlib = openrlib.rlib
    if val is None:
        s = rlib.R_NaString
    else:
        cchar = _str_to_cchar(val, encoding='utf-8')
        s = rlib.Rf_mkCharCE(cchar, openrlib.rlib.CE_UTF8)
    return s