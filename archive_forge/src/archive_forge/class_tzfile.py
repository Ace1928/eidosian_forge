from relative deltas), local machine timezone, fixed offset timezone, and UTC
import datetime
import logging  # GOOGLE
import struct
import time
import sys
import os
import bisect
import weakref
from collections import OrderedDict
import six
from six import string_types
from six.moves import _thread
from ._common import tzname_in_python2, _tzinfo
from ._common import tzrangebase, enfold
from ._common import _validate_fromutc_inputs
from ._factories import _TzSingleton, _TzOffsetFactory
from ._factories import _TzStrFactory
from warnings import warn
class tzfile(_tzinfo):
    """
    This is a ``tzinfo`` subclass that allows one to use the ``tzfile(5)``
    format timezone files to extract current and historical zone information.

    :param fileobj:
        This can be an opened file stream or a file name that the time zone
        information can be read from.

    :param filename:
        This is an optional parameter specifying the source of the time zone
        information in the event that ``fileobj`` is a file object. If omitted
        and ``fileobj`` is a file stream, this parameter will be set either to
        ``fileobj``'s ``name`` attribute or to ``repr(fileobj)``.

    See `Sources for Time Zone and Daylight Saving Time Data
    <https://data.iana.org/time-zones/tz-link.html>`_ for more information.
    Time zone files can be compiled from the `IANA Time Zone database files
    <https://www.iana.org/time-zones>`_ with the `zic time zone compiler
    <https://www.freebsd.org/cgi/man.cgi?query=zic&sektion=8>`_

    .. note::

        Only construct a ``tzfile`` directly if you have a specific timezone
        file on disk that you want to read into a Python ``tzinfo`` object.
        If you want to get a ``tzfile`` representing a specific IANA zone,
        (e.g. ``'America/New_York'``), you should call
        :func:`dateutil.tz.gettz` with the zone identifier.


    **Examples:**

    Using the US Eastern time zone as an example, we can see that a ``tzfile``
    provides time zone information for the standard Daylight Saving offsets:

    .. testsetup:: tzfile

        from dateutil.tz import gettz
        from datetime import datetime

    .. doctest:: tzfile

        >>> NYC = gettz('America/New_York')
        >>> NYC
        tzfile('/usr/share/zoneinfo/America/New_York')

        >>> print(datetime(2016, 1, 3, tzinfo=NYC))     # EST
        2016-01-03 00:00:00-05:00

        >>> print(datetime(2016, 7, 7, tzinfo=NYC))     # EDT
        2016-07-07 00:00:00-04:00


    The ``tzfile`` structure contains a fully history of the time zone,
    so historical dates will also have the right offsets. For example, before
    the adoption of the UTC standards, New York used local solar  mean time:

    .. doctest:: tzfile

       >>> print(datetime(1901, 4, 12, tzinfo=NYC))    # LMT
       1901-04-12 00:00:00-04:56

    And during World War II, New York was on "Eastern War Time", which was a
    state of permanent daylight saving time:

    .. doctest:: tzfile

        >>> print(datetime(1944, 2, 7, tzinfo=NYC))    # EWT
        1944-02-07 00:00:00-04:00

    """

    def __init__(self, fileobj, filename=None):
        super(tzfile, self).__init__()
        file_opened_here = False
        if isinstance(fileobj, string_types):
            self._filename = fileobj
            fileobj = open(fileobj, 'rb')
            file_opened_here = True
        elif filename is not None:
            self._filename = filename
        elif hasattr(fileobj, 'name'):
            self._filename = fileobj.name
        else:
            self._filename = repr(fileobj)
        if fileobj is not None:
            if not file_opened_here:
                fileobj = _nullcontext(fileobj)
            with fileobj as file_stream:
                tzobj = self._read_tzfile(file_stream)
            self._set_tzdata(tzobj)

    def _set_tzdata(self, tzobj):
        """ Set the time zone data of this object from a _tzfile object """
        for attr in _tzfile.attrs:
            setattr(self, '_' + attr, getattr(tzobj, attr))

    def _read_tzfile(self, fileobj):
        out = _tzfile()
        if fileobj.read(4).decode() != 'TZif':
            raise ValueError('magic not found')
        fileobj.read(16)
        ttisgmtcnt, ttisstdcnt, leapcnt, timecnt, typecnt, charcnt = struct.unpack('>6l', fileobj.read(24))
        if timecnt:
            out.trans_list_utc = list(struct.unpack('>%dl' % timecnt, fileobj.read(timecnt * 4)))
        else:
            out.trans_list_utc = []
        if timecnt:
            out.trans_idx = struct.unpack('>%dB' % timecnt, fileobj.read(timecnt))
        else:
            out.trans_idx = []
        ttinfo = []
        for i in range(typecnt):
            ttinfo.append(struct.unpack('>lbb', fileobj.read(6)))
        abbr = fileobj.read(charcnt).decode()
        if leapcnt:
            fileobj.seek(leapcnt * 8, os.SEEK_CUR)
        if ttisstdcnt:
            isstd = struct.unpack('>%db' % ttisstdcnt, fileobj.read(ttisstdcnt))
        if ttisgmtcnt:
            isgmt = struct.unpack('>%db' % ttisgmtcnt, fileobj.read(ttisgmtcnt))
        out.ttinfo_list = []
        for i in range(typecnt):
            gmtoff, isdst, abbrind = ttinfo[i]
            gmtoff = _get_supported_offset(gmtoff)
            tti = _ttinfo()
            tti.offset = gmtoff
            tti.dstoffset = datetime.timedelta(0)
            tti.delta = datetime.timedelta(seconds=gmtoff)
            tti.isdst = isdst
            tti.abbr = abbr[abbrind:abbr.find('\x00', abbrind)]
            tti.isstd = ttisstdcnt > i and isstd[i] != 0
            tti.isgmt = ttisgmtcnt > i and isgmt[i] != 0
            out.ttinfo_list.append(tti)
        out.trans_idx = [out.ttinfo_list[idx] for idx in out.trans_idx]
        out.ttinfo_std = None
        out.ttinfo_dst = None
        out.ttinfo_before = None
        if out.ttinfo_list:
            if not out.trans_list_utc:
                out.ttinfo_std = out.ttinfo_first = out.ttinfo_list[0]
            else:
                for i in range(timecnt - 1, -1, -1):
                    tti = out.trans_idx[i]
                    if not out.ttinfo_std and (not tti.isdst):
                        out.ttinfo_std = tti
                    elif not out.ttinfo_dst and tti.isdst:
                        out.ttinfo_dst = tti
                    if out.ttinfo_std and out.ttinfo_dst:
                        break
                else:
                    if out.ttinfo_dst and (not out.ttinfo_std):
                        out.ttinfo_std = out.ttinfo_dst
                for tti in out.ttinfo_list:
                    if not tti.isdst:
                        out.ttinfo_before = tti
                        break
                else:
                    out.ttinfo_before = out.ttinfo_list[0]
        lastdst = None
        lastoffset = None
        lastdstoffset = None
        lastbaseoffset = None
        out.trans_list = []
        for i, tti in enumerate(out.trans_idx):
            offset = tti.offset
            dstoffset = 0
            if lastdst is not None:
                if tti.isdst:
                    if not lastdst:
                        dstoffset = offset - lastoffset
                    if not dstoffset and lastdstoffset:
                        dstoffset = lastdstoffset
                    tti.dstoffset = datetime.timedelta(seconds=dstoffset)
                    lastdstoffset = dstoffset
            baseoffset = offset - dstoffset
            adjustment = baseoffset
            if lastbaseoffset is not None and baseoffset != lastbaseoffset and (tti.isdst != lastdst):
                adjustment = lastbaseoffset
            lastdst = tti.isdst
            lastoffset = offset
            lastbaseoffset = baseoffset
            out.trans_list.append(out.trans_list_utc[i] + adjustment)
        out.trans_idx = tuple(out.trans_idx)
        out.trans_list = tuple(out.trans_list)
        out.trans_list_utc = tuple(out.trans_list_utc)
        return out

    def _find_last_transition(self, dt, in_utc=False):
        if not self._trans_list:
            return None
        timestamp = _datetime_to_timestamp(dt)
        trans_list = self._trans_list_utc if in_utc else self._trans_list
        idx = bisect.bisect_right(trans_list, timestamp)
        return idx - 1

    def _get_ttinfo(self, idx):
        if idx is None or idx + 1 >= len(self._trans_list):
            return self._ttinfo_std
        if idx < 0:
            return self._ttinfo_before
        return self._trans_idx[idx]

    def _find_ttinfo(self, dt):
        idx = self._resolve_ambiguous_time(dt)
        return self._get_ttinfo(idx)

    def fromutc(self, dt):
        """
        The ``tzfile`` implementation of :py:func:`datetime.tzinfo.fromutc`.

        :param dt:
            A :py:class:`datetime.datetime` object.

        :raises TypeError:
            Raised if ``dt`` is not a :py:class:`datetime.datetime` object.

        :raises ValueError:
            Raised if this is called with a ``dt`` which does not have this
            ``tzinfo`` attached.

        :return:
            Returns a :py:class:`datetime.datetime` object representing the
            wall time in ``self``'s time zone.
        """
        if not isinstance(dt, datetime.datetime):
            raise TypeError('fromutc() requires a datetime argument')
        if dt.tzinfo is not self:
            raise ValueError('dt.tzinfo is not self')
        idx = self._find_last_transition(dt, in_utc=True)
        tti = self._get_ttinfo(idx)
        dt_out = dt + datetime.timedelta(seconds=tti.offset)
        fold = self.is_ambiguous(dt_out, idx=idx)
        return enfold(dt_out, fold=int(fold))

    def is_ambiguous(self, dt, idx=None):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        if idx is None:
            idx = self._find_last_transition(dt)
        timestamp = _datetime_to_timestamp(dt)
        tti = self._get_ttinfo(idx)
        if idx is None or idx <= 0:
            return False
        od = self._get_ttinfo(idx - 1).offset - tti.offset
        tt = self._trans_list[idx]
        return timestamp < tt + od

    def _resolve_ambiguous_time(self, dt):
        idx = self._find_last_transition(dt)
        _fold = self._fold(dt)
        if idx is None or idx == 0:
            return idx
        idx_offset = int(not _fold and self.is_ambiguous(dt, idx))
        return idx - idx_offset

    def utcoffset(self, dt):
        if dt is None:
            return None
        if not self._ttinfo_std:
            return ZERO
        return self._find_ttinfo(dt).delta

    def dst(self, dt):
        if dt is None:
            return None
        if not self._ttinfo_dst:
            return ZERO
        tti = self._find_ttinfo(dt)
        if not tti.isdst:
            return ZERO
        return tti.dstoffset

    @tzname_in_python2
    def tzname(self, dt):
        if not self._ttinfo_std or dt is None:
            return None
        return self._find_ttinfo(dt).abbr

    def __eq__(self, other):
        if not isinstance(other, tzfile):
            return NotImplemented
        return self._trans_list == other._trans_list and self._trans_idx == other._trans_idx and (self._ttinfo_list == other._ttinfo_list)
    __hash__ = None

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self._filename))

    def __reduce__(self):
        return self.__reduce_ex__(None)

    def __reduce_ex__(self, protocol):
        return (self.__class__, (None, self._filename), self.__dict__)