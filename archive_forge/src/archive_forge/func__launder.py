from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
def _launder(x):
    """ Clean up types prior to insertion into DataShape

    >>> from datashader.datashape import dshape
    >>> _launder(5)         # convert ints to Fixed
    Fixed(val=5)
    >>> _launder('int32')   # parse strings
    ctype("int32")
    >>> _launder(dshape('int32'))
    ctype("int32")
    >>> _launder(Fixed(5))  # No-op on valid parameters
    Fixed(val=5)
    """
    if isinstance(x, int):
        x = Fixed(x)
    if isinstance(x, str):
        x = datashape.dshape(x)
    if isinstance(x, DataShape) and len(x) == 1:
        return x[0]
    if isinstance(x, Mono):
        return x
    return x