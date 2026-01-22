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
def _py2rpy(obj):
    """ Dummy function for py2rpy.

    This function will convert Python objects into rpy2.rinterface
    objects.
    """
    if isinstance(obj, _rinterface_capi.SupportsSEXP):
        return obj
    raise NotImplementedError("Conversion 'py2rpy' not defined for objects of type '%s'" % str(type(obj)))