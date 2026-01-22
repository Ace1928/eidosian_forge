import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class TimeDependent(TimeAware):
    """
    Objects that have access to a time function that determines the
    output value. As a function of time, this type of object should
    allow time values to be randomly jumped forwards or backwards,
    but for a given time point, the results should remain constant.

    The time_fn must be an instance of param.Time, to ensure all the
    facilities necessary for safely navigating the timeline are
    available.
    """
    time_dependent = param.Boolean(default=True, readonly=True, doc='\n       Read-only parameter that is always True.')

    def _check_time_fn(self):
        super()._check_time_fn(time_instance=True)