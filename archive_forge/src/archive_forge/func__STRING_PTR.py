import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def _STRING_PTR(robj):
    return ffi.cast('SEXP *', DATAPTR(robj))