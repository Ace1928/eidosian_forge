import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
class POSIXlt(POSIXt, ListVector):
    """ Representation of dates with a 11-component structure
    (similar to Python's time.struct_time).

    POSIXlt(seq) -> POSIXlt.

    The constructor accepts either an R vector
    or a sequence (an object with the Python
    sequence interface) of time.struct_time objects.
    """
    _expected_colnames = {x: i for i, x in enumerate(('sec', 'min', 'hour', 'mday', 'mon', 'year', 'wday', 'yday', 'isdst', 'zone', 'gmtoff'))}
    _wday_r_to_py = (6, 0, 1, 2, 3, 4, 5)

    def __init__(self, seq):
        """
        """
        if isinstance(seq, Sexp):
            super()(seq)
        else:
            for elt in conversion.noconversion(seq):
                if not isinstance(elt, struct_time):
                    raise ValueError('All elements must inherit from time.struct_time')
            as_posixlt = baseenv_ri['as.POSIXlt']
            origin = StrSexpVector([time.strftime('%Y-%m-%d', time.gmtime(0))])
            rvec = FloatSexpVector([mktime(x) for x in seq])
            sexp = as_posixlt(rvec, origin=origin)
            super().__init__(sexp)

    def __getitem__(self, i):
        aslist = ListVector(self)
        idx = self._expected_colnames
        seq = (aslist[idx['year']][i] + 1900, aslist[idx['mon']][i] + 1, aslist[idx['mday']][i], aslist[idx['hour']][i], aslist[idx['min']][i], aslist[idx['sec']][i], self._wday_r_to_py[aslist[idx['wday']][i]], aslist[idx['yday']][i] + 1, aslist[idx['isdst']][i])
        return struct_time(seq, {'tm_zone': aslist[idx['zone']][i], 'tmp_gmtoff': aslist[idx['gmtoff']][i]})

    def __repr__(self):
        return super(Sexp, self).__repr__()