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
def _assert_valid_slotname(name: str) -> None:
    if not isinstance(name, str):
        raise ValueError('Invalid name %s' % repr(name))
    elif len(name) == 0:
        raise ValueError('The name cannot be an empty string')