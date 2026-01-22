import inspect
import os
import re
import textwrap
import typing
from typing import Union
import warnings
from collections import OrderedDict
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp
from rpy2.robjects import help
from rpy2.robjects import conversion
from rpy2.robjects.vectors import Vector
from rpy2.robjects.packages_utils import (default_symbol_r2python,
def _repr_argval(obj):
    """ Helper functions to display an R object in the docstring.
    This a hack and will be hopefully replaced the extraction of
    information from the R help system."""
    try:
        size = len(obj)
        if size == 1:
            if obj[0].rid == rinterface.MissingArg.rid:
                s = None
            elif obj[0].rid == rinterface.NULL.rid:
                s = 'rinterface.NULL'
            else:
                s = str(obj[0][0])
        elif size > 1:
            s = '(%s, ...)' % str(obj[0][0])
        else:
            s = str(obj)
    except Exception:
        s = str(obj)
    return s