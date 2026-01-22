from six import PY2
from functools import wraps
from datetime import datetime, timedelta, tzinfo
class tzrangebase(_tzinfo):
    """
    This is an abstract base class for time zones represented by an annual
    transition into and out of DST. Child classes should implement the following
    methods:

        * ``__init__(self, *args, **kwargs)``
        * ``transitions(self, year)`` - this is expected to return a tuple of
          datetimes representing the DST on and off transitions in standard
          time.

    A fully initialized ``tzrangebase`` subclass should also provide the
    following attributes:
        * ``hasdst``: Boolean whether or not the zone uses DST.
        * ``_dst_offset`` / ``_std_offset``: :class:`datetime.timedelta` objects
          representing the respective UTC offsets.
        * ``_dst_abbr`` / ``_std_abbr``: Strings representing the timezone short
          abbreviations in DST and STD, respectively.
        * ``_hasdst``: Whether or not the zone has DST.

    .. versionadded:: 2.6.0
    """

    def __init__(self):
        raise NotImplementedError('tzrangebase is an abstract base class')

    def utcoffset(self, dt):
        isdst = self._isdst(dt)
        if isdst is None:
            return None
        elif isdst:
            return self._dst_offset
        else:
            return self._std_offset

    def dst(self, dt):
        isdst = self._isdst(dt)
        if isdst is None:
            return None
        elif isdst:
            return self._dst_base_offset
        else:
            return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        if self._isdst(dt):
            return self._dst_abbr
        else:
            return self._std_abbr

    def fromutc(self, dt):
        """ Given a datetime in UTC, return local time """
        if not isinstance(dt, datetime):
            raise TypeError('fromutc() requires a datetime argument')
        if dt.tzinfo is not self:
            raise ValueError('dt.tzinfo is not self')
        transitions = self.transitions(dt.year)
        if transitions is None:
            return dt + self.utcoffset(dt)
        dston, dstoff = transitions
        dston -= self._std_offset
        dstoff -= self._std_offset
        utc_transitions = (dston, dstoff)
        dt_utc = dt.replace(tzinfo=None)
        isdst = self._naive_isdst(dt_utc, utc_transitions)
        if isdst:
            dt_wall = dt + self._dst_offset
        else:
            dt_wall = dt + self._std_offset
        _fold = int(not isdst and self.is_ambiguous(dt_wall))
        return enfold(dt_wall, fold=_fold)

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        if not self.hasdst:
            return False
        start, end = self.transitions(dt.year)
        dt = dt.replace(tzinfo=None)
        return end <= dt < end + self._dst_base_offset

    def _isdst(self, dt):
        if not self.hasdst:
            return False
        elif dt is None:
            return None
        transitions = self.transitions(dt.year)
        if transitions is None:
            return False
        dt = dt.replace(tzinfo=None)
        isdst = self._naive_isdst(dt, transitions)
        if not isdst and self.is_ambiguous(dt):
            return not self._fold(dt)
        else:
            return isdst

    def _naive_isdst(self, dt, transitions):
        dston, dstoff = transitions
        dt = dt.replace(tzinfo=None)
        if dston < dstoff:
            isdst = dston <= dt < dstoff
        else:
            isdst = not dstoff <= dt < dston
        return isdst

    @property
    def _dst_base_offset(self):
        return self._dst_offset - self._std_offset
    __hash__ = None

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '%s(...)' % self.__class__.__name__
    __reduce__ = object.__reduce__