import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def _get_complex_elt_fallback(vec, i: int):
    return COMPLEX(vec)[i]