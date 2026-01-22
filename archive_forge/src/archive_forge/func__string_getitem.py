import abc
import enum
import inspect
import logging
from typing import Tuple
import typing
import warnings
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from _cffi_backend import FFI  # type: ignore
def _string_getitem(cdata: FFI.CData, i: int) -> typing.Optional[str]:
    elt = openrlib.rlib.STRING_ELT(cdata, i)
    if elt == openrlib.rlib.R_NaString:
        res = None
    else:
        res = conversion._cchar_to_str(openrlib.rlib.R_CHAR(elt), conversion._R_ENC_PY[openrlib.rlib.Rf_getCharCE(elt)])
    return res