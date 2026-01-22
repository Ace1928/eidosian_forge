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
class PARSING_STATUS(enum.Enum):
    PARSE_NULL = openrlib.rlib.PARSE_NULL
    PARSE_OK = openrlib.rlib.PARSE_OK
    PARSE_INCOMPLETE = openrlib.rlib.PARSE_INCOMPLETE
    PARSE_ERROR = openrlib.rlib.PARSE_ERROR
    PARSE_EOF = openrlib.rlib.PARSE_EOF