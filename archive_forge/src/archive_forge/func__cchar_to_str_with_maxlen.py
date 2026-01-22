from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _cchar_to_str_with_maxlen(c, maxlen: int, encoding: str) -> str:
    s = ffi.string(c, maxlen).decode(encoding)
    return s