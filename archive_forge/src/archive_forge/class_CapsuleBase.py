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
class CapsuleBase:
    _cdata: FFI.CData

    @property
    def typeof(self) -> int:
        return _TYPEOF(self._cdata)

    @property
    def rid(self) -> int:
        return get_rid(self._cdata)