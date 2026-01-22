import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def _COMPLEX(robj):
    return ffi.cast('Rcomplex *', DATAPTR(robj))