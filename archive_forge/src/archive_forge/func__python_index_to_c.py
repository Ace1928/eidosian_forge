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
def _python_index_to_c(cdata: FFI.CData, i: int) -> int:
    """Compute the C value for a Python-style index.

    The function will translate a Python-style index that
    can be either positive or negative, if the later to
    indicate that indexing is relative to the end of the
    sequence, into a strictly positive or null index as
    use in C.

    Raises an exception IndexError if out of bounds."""
    size = openrlib.rlib.Rf_xlength(cdata)
    if i < 0:
        i = size + i
    if i >= size or i < 0:
        raise IndexError('index out of range')
    return i