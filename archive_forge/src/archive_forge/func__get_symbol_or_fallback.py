import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def _get_symbol_or_fallback(symbol: str, fallback: typing.Any):
    """Get a cffi object from rlib, or the fallback if missing."""
    try:
        res = getattr(rlib, symbol)
    except (ffi.error, AttributeError):
        res = fallback
    return res