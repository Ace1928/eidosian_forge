import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def _join_by_dispatcher(key, r1, r2, jointype=None, r1postfix=None, r2postfix=None, defaults=None, usemask=None, asrecarray=None):
    return (r1, r2)