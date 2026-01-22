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
class POSIXct(POSIXt, FloatVector):
    """ Representation of dates as seconds since Epoch.
    This form is preferred to POSIXlt for inclusion in a DataFrame.

    POSIXlt(seq) -> POSIXlt.

    The constructor accepts either an R vector floats
    or a sequence (an object with the Python
    sequence interface) of time.struct_time objects.
    """
    _as_posixct = baseenv_ri['as.POSIXct']
    _ISOdatetime = baseenv_ri['ISOdatetime']

    def __init__(self, seq):
        """ Create a POSIXct from either an R vector or a sequence
        of Python dates.
        """
        if isinstance(seq, Sexp):
            init_param = seq
        elif isinstance(seq[0], struct_time):
            init_param = POSIXct.sexp_from_struct_time(seq)
        elif isinstance(seq[0], datetime.datetime):
            init_param = POSIXct.sexp_from_datetime(seq)
        else:
            raise TypeError('All elements must inherit from time.struct_time or datetime.datetime.')
        super().__init__(init_param)

    @staticmethod
    def _sexp_from_seq(seq, tz_info_getter, isodatetime_columns):
        """ return a POSIXct vector from a sequence of time.struct_time
        elements. """
        tz_count = 0
        tz_info = None
        for elt in conversion.noconversion(seq):
            tmp = tz_info_getter(elt)
            if tz_info is None:
                tz_info = tmp
                tz_count = 1
            elif tz_info == tmp:
                tz_count += 1
            else:
                raise ValueError('Sequences of dates with different time zones not yet allowed.')
        if tz_info is None:
            tz_info = default_timezone if default_timezone else ''
        d = isodatetime_columns(seq)
        sexp = POSIXct._ISOdatetime(*d, tz=StrSexpVector((str(tz_info),)))
        return sexp

    @staticmethod
    def sexp_from_struct_time(seq):

        def f(seq):
            return [IntVector([x.tm_year for x in seq]), IntVector([x.tm_mon for x in seq]), IntVector([x.tm_mday for x in seq]), IntVector([x.tm_hour for x in seq]), IntVector([x.tm_min for x in seq]), IntVector([x.tm_sec for x in seq])]
        return POSIXct._sexp_from_seq(seq, lambda elt: elt.tm_zone, f)

    @staticmethod
    def sexp_from_datetime(seq):
        """ return a POSIXct vector from a sequence of
        datetime.datetime elements. """

        def f(seq):
            return [IntVector([x.year for x in seq]), IntVector([x.month for x in seq]), IntVector([x.day for x in seq]), IntVector([x.hour for x in seq]), IntVector([x.minute for x in seq]), IntVector([x.second for x in seq])]

        def get_tz(elt):
            return elt.tzinfo if elt.tzinfo else None
        return POSIXct._sexp_from_seq(seq, get_tz, f)

    @staticmethod
    def isrinstance(obj) -> bool:
        """Is an R object an instance of POSIXct."""
        return obj.rclass[0] == 'POSIXct'

    @staticmethod
    def _datetime_from_timestamp(ts, tz) -> datetime.datetime:
        """Platform-dependent conversion from timestamp to datetime"""
        if os.name != 'nt' or ts > 0:
            return datetime.datetime.fromtimestamp(ts, tz)
        else:
            dt_utc = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=ts)
            dt = dt_utc.replace(tzinfo=tz)
            offset = dt.utcoffset()
            if offset is None:
                return dt
            else:
                return dt + offset

    def iter_localized_datetime(self):
        """Iterator yielding localized Python datetime objects."""
        try:
            r_tzone_name = self.do_slot('tzone')[0]
        except LookupError:
            warnings.warn('R object inheriting from "POSIXct" but without attribute "tzone".')
            r_tzone_name = ''
        if r_tzone_name == '':
            r_tzone = get_timezone()
        else:
            r_tzone = zoneinfo.ZoneInfo(r_tzone_name)
        for x in self:
            yield (None if math.isnan(x) else POSIXct._datetime_from_timestamp(x, r_tzone))