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
class UninitializedRCapsule(CapsuleBase):

    @property
    def _cdata(self):
        raise RuntimeError('The embedded R is not initialized.')

    @property
    def typeof(self) -> int:
        return self._typeof

    def __init__(self, typeof):
        self._typeof = typeof