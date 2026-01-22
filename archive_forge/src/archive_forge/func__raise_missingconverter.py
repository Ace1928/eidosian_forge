import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
def _raise_missingconverter(obj):
    _missingconverter_msg = "\n    Conversion rules for `rpy2.robjects` appear to be missing. Those\n    rules are in a Python `contextvars.ContextVar`. This could be caused\n    by multithreading code not passing context to the thread.\n    Check rpy2's documentation about conversions.\n    "
    raise NotImplementedError(_missingconverter_msg)