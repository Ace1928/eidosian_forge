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
class tzical(object):
    """
    This object is designed to parse an iCalendar-style ``VTIMEZONE`` structure
    as set out in `RFC 5545`_ Section 4.6.5 into one or more `tzinfo` objects.

    :param `fileobj`:
        A file or stream in iCalendar format, which should be UTF-8 encoded
        with CRLF endings.

    .. _`RFC 5545`: https://tools.ietf.org/html/rfc5545
    """

    def __init__(self, fileobj):
        global rrule
        from dateutil import rrule
        if isinstance(fileobj, string_types):
            self._s = fileobj
            fileobj = open(fileobj, 'r')
        else:
            self._s = getattr(fileobj, 'name', repr(fileobj))
            fileobj = _nullcontext(fileobj)
        self._vtz = {}
        with fileobj as fobj:
            self._parse_rfc(fobj.read())

    def keys(self):
        """
        Retrieves the available time zones as a list.
        """
        return list(self._vtz.keys())

    def get(self, tzid=None):
        """
        Retrieve a :py:class:`datetime.tzinfo` object by its ``tzid``.

        :param tzid:
            If there is exactly one time zone available, omitting ``tzid``
            or passing :py:const:`None` value returns it. Otherwise a valid
            key (which can be retrieved from :func:`keys`) is required.

        :raises ValueError:
            Raised if ``tzid`` is not specified but there are either more
            or fewer than 1 zone defined.

        :returns:
            Returns either a :py:class:`datetime.tzinfo` object representing
            the relevant time zone or :py:const:`None` if the ``tzid`` was
            not found.
        """
        if tzid is None:
            if len(self._vtz) == 0:
                raise ValueError('no timezones defined')
            elif len(self._vtz) > 1:
                raise ValueError('more than one timezone available')
            tzid = next(iter(self._vtz))
        return self._vtz.get(tzid)

    def _parse_offset(self, s):
        s = s.strip()
        if not s:
            raise ValueError('empty offset')
        if s[0] in ('+', '-'):
            signal = (-1, +1)[s[0] == '+']
            s = s[1:]
        else:
            signal = +1
        if len(s) == 4:
            return (int(s[:2]) * 3600 + int(s[2:]) * 60) * signal
        elif len(s) == 6:
            return (int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:])) * signal
        else:
            raise ValueError('invalid offset: ' + s)

    def _parse_rfc(self, s):
        lines = s.splitlines()
        if not lines:
            raise ValueError('empty string')
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line:
                del lines[i]
            elif i > 0 and line[0] == ' ':
                lines[i - 1] += line[1:]
                del lines[i]
            else:
                i += 1
        tzid = None
        comps = []
        invtz = False
        comptype = None
        for line in lines:
            if not line:
                continue
            name, value = line.split(':', 1)
            parms = name.split(';')
            if not parms:
                raise ValueError('empty property name')
            name = parms[0].upper()
            parms = parms[1:]
            if invtz:
                if name == 'BEGIN':
                    if value in ('STANDARD', 'DAYLIGHT'):
                        pass
                    else:
                        raise ValueError('unknown component: ' + value)
                    comptype = value
                    founddtstart = False
                    tzoffsetfrom = None
                    tzoffsetto = None
                    rrulelines = []
                    tzname = None
                elif name == 'END':
                    if value == 'VTIMEZONE':
                        if comptype:
                            raise ValueError('component not closed: ' + comptype)
                        if not tzid:
                            raise ValueError('mandatory TZID not found')
                        if not comps:
                            raise ValueError('at least one component is needed')
                        self._vtz[tzid] = _tzicalvtz(tzid, comps)
                        invtz = False
                    elif value == comptype:
                        if not founddtstart:
                            raise ValueError('mandatory DTSTART not found')
                        if tzoffsetfrom is None:
                            raise ValueError('mandatory TZOFFSETFROM not found')
                        if tzoffsetto is None:
                            raise ValueError('mandatory TZOFFSETFROM not found')
                        rr = None
                        if rrulelines:
                            rr = rrule.rrulestr('\n'.join(rrulelines), compatible=True, ignoretz=True, cache=True)
                        comp = _tzicalvtzcomp(tzoffsetfrom, tzoffsetto, comptype == 'DAYLIGHT', tzname, rr)
                        comps.append(comp)
                        comptype = None
                    else:
                        raise ValueError('invalid component end: ' + value)
                elif comptype:
                    if name == 'DTSTART':
                        for parm in parms:
                            if parm != 'VALUE=DATE-TIME':
                                msg = 'Unsupported DTSTART param in ' + 'VTIMEZONE: ' + parm
                                raise ValueError(msg)
                        rrulelines.append(line)
                        founddtstart = True
                    elif name in ('RRULE', 'RDATE', 'EXRULE', 'EXDATE'):
                        rrulelines.append(line)
                    elif name == 'TZOFFSETFROM':
                        if parms:
                            raise ValueError('unsupported %s parm: %s ' % (name, parms[0]))
                        tzoffsetfrom = self._parse_offset(value)
                    elif name == 'TZOFFSETTO':
                        if parms:
                            raise ValueError('unsupported TZOFFSETTO parm: ' + parms[0])
                        tzoffsetto = self._parse_offset(value)
                    elif name == 'TZNAME':
                        if parms:
                            raise ValueError('unsupported TZNAME parm: ' + parms[0])
                        tzname = value
                    elif name == 'COMMENT':
                        pass
                    else:
                        raise ValueError('unsupported property: ' + name)
                elif name == 'TZID':
                    if parms:
                        raise ValueError('unsupported TZID parm: ' + parms[0])
                    tzid = value
                elif name in ('TZURL', 'LAST-MODIFIED', 'COMMENT'):
                    pass
                else:
                    raise ValueError('unsupported property: ' + name)
            elif name == 'BEGIN' and value == 'VTIMEZONE':
                tzid = None
                comps = []
                invtz = True

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self._s))