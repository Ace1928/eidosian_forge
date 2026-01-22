import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def _INTEGER(x):
    return ffi.cast('int *', DATAPTR(x))