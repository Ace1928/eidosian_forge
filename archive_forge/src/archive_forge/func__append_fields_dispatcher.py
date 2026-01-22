import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def _append_fields_dispatcher(base, names, data, dtypes=None, fill_value=None, usemask=None, asrecarray=None):
    yield base
    yield from data