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
class DateVector(FloatVector):
    """ Representation of dates as number of days since 1/1/1970.

    Date(seq) -> Date.

    The constructor accepts either an R vector floats
    or a sequence (an object with the Python
    sequence interface) of time.struct_time objects.
    """

    def __init__(self, seq):
        """ Create a POSIXct from either an R vector or a sequence
        of Python datetime.date objects.
        """
        if isinstance(seq, Sexp):
            init_param = seq
        elif isinstance(seq[0], datetime.date):
            init_param = DateVector.sexp_from_date(seq)
        else:
            raise TypeError('Unable to create an R Date vector from objects of type %s' % type(seq))
        super().__init__(init_param)

    @classmethod
    def sexp_from_date(cls, seq):
        return cls(FloatVector([x.toordinal() for x in seq]))

    @staticmethod
    def isrinstance(obj) -> bool:
        """Return whether an R object an instance of Date."""
        return obj.rclass[-1] == 'Date'