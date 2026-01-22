import collections.abc
import contextlib
import contextvars
from .._utils import set_module
from .umath import (
from . import umath
def _setdef():
    defval = [UFUNC_BUFSIZE_DEFAULT, ERR_DEFAULT, None]
    umath.seterrobj(defval)