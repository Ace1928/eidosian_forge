import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def _setcallback(rlib, rlib_symbol: str, callbacks, callback_symbol: typing.Optional[str]) -> None:
    """Set R callbacks."""
    if callback_symbol is None:
        new_callback = ffi.NULL
    else:
        new_callback = getattr(callbacks, callback_symbol)
    setattr(rlib, rlib_symbol, new_callback)