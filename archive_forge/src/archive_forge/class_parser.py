from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
class parser(object):

    def __init__(self, info=None):
        self.info = info or parserinfo()

    def parse(self, timestr, default=None, ignoretz=False, tzinfos=None, **kwargs):
        """
        Parse the date/time string into a :class:`datetime.datetime` object.

        :param timestr:
            Any date/time string using the supported formats.

        :param default:
            The default datetime object, if this is a datetime object and not
            ``None``, elements specified in ``timestr`` replace elements in the
            default object.

        :param ignoretz:
            If set ``True``, time zones in parsed strings are ignored and a
            naive :class:`datetime.datetime` object is returned.

        :param tzinfos:
            Additional time zone names / aliases which may be present in the
            string. This argument maps time zone names (and optionally offsets
            from those time zones) to time zones. This parameter can be a
            dictionary with timezone aliases mapping time zone names to time
            zones or a function taking two parameters (``tzname`` and
            ``tzoffset``) and returning a time zone.

            The timezones to which the names are mapped can be an integer
            offset from UTC in seconds or a :class:`tzinfo` object.

            .. doctest::
               :options: +NORMALIZE_WHITESPACE

                >>> from dateutil.parser import parse
                >>> from dateutil.tz import gettz
                >>> tzinfos = {"BRST": -7200, "CST": gettz("America/Chicago")}
                >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -7200))
                >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21,
                                  tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

            This parameter is ignored if ``ignoretz`` is set.

        :param \\*\\*kwargs:
            Keyword arguments as passed to ``_parse()``.

        :return:
            Returns a :class:`datetime.datetime` object or, if the
            ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
            first element being a :class:`datetime.datetime` object, the second
            a tuple containing the fuzzy tokens.

        :raises ParserError:
            Raised for invalid or unknown string format, if the provided
            :class:`tzinfo` is not in a valid format, or if an invalid date
            would be created.

        :raises TypeError:
            Raised for non-string or character stream input.

        :raises OverflowError:
            Raised if the parsed date exceeds the largest valid C integer on
            your system.
        """
        if default is None:
            default = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        res, skipped_tokens = self._parse(timestr, **kwargs)
        if res is None:
            raise ParserError('Unknown string format: %s', timestr)
        if len(res) == 0:
            raise ParserError('String does not contain a date: %s', timestr)
        try:
            ret = self._build_naive(res, default)
        except ValueError as e:
            six.raise_from(ParserError(str(e) + ': %s', timestr), e)
        if not ignoretz:
            ret = self._build_tzaware(ret, res, tzinfos)
        if kwargs.get('fuzzy_with_tokens', False):
            return (ret, skipped_tokens)
        else:
            return ret

    class _result(_resultbase):
        __slots__ = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second', 'microsecond', 'tzname', 'tzoffset', 'ampm', 'any_unused_tokens']

    def _parse(self, timestr, dayfirst=None, yearfirst=None, fuzzy=False, fuzzy_with_tokens=False):
        """
        Private method which performs the heavy lifting of parsing, called from
        ``parse()``, which passes on its ``kwargs`` to this function.

        :param timestr:
            The string to parse.

        :param dayfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the day (``True``) or month (``False``). If
            ``yearfirst`` is set to ``True``, this distinguishes between YDM
            and YMD. If set to ``None``, this value is retrieved from the
            current :class:`parserinfo` object (which itself defaults to
            ``False``).

        :param yearfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the year. If ``True``, the first number is taken
            to be the year, otherwise the last number is taken to be the year.
            If this is set to ``None``, the value is retrieved from the current
            :class:`parserinfo` object (which itself defaults to ``False``).

        :param fuzzy:
            Whether to allow fuzzy parsing, allowing for string like "Today is
            January 1, 2047 at 8:21:00AM".

        :param fuzzy_with_tokens:
            If ``True``, ``fuzzy`` is automatically set to True, and the parser
            will return a tuple where the first element is the parsed
            :class:`datetime.datetime` datetimestamp and the second element is
            a tuple containing the portions of the string which were ignored:

            .. doctest::

                >>> from dateutil.parser import parse
                >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
                (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

        """
        if fuzzy_with_tokens:
            fuzzy = True
        info = self.info
        if dayfirst is None:
            dayfirst = info.dayfirst
        if yearfirst is None:
            yearfirst = info.yearfirst
        res = self._result()
        l = _timelex.split(timestr)
        skipped_idxs = []
        ymd = _ymd()
        len_l = len(l)
        i = 0
        try:
            while i < len_l:
                value_repr = l[i]
                try:
                    value = float(value_repr)
                except ValueError:
                    value = None
                if value is not None:
                    i = self._parse_numeric_token(l, i, info, ymd, res, fuzzy)
                elif info.weekday(l[i]) is not None:
                    value = info.weekday(l[i])
                    res.weekday = value
                elif info.month(l[i]) is not None:
                    value = info.month(l[i])
                    ymd.append(value, 'M')
                    if i + 1 < len_l:
                        if l[i + 1] in ('-', '/'):
                            sep = l[i + 1]
                            ymd.append(l[i + 2])
                            if i + 3 < len_l and l[i + 3] == sep:
                                ymd.append(l[i + 4])
                                i += 2
                            i += 2
                        elif i + 4 < len_l and l[i + 1] == l[i + 3] == ' ' and info.pertain(l[i + 2]):
                            if l[i + 4].isdigit():
                                value = int(l[i + 4])
                                year = str(info.convertyear(value))
                                ymd.append(year, 'Y')
                            else:
                                pass
                            i += 4
                elif info.ampm(l[i]) is not None:
                    value = info.ampm(l[i])
                    val_is_ampm = self._ampm_valid(res.hour, res.ampm, fuzzy)
                    if val_is_ampm:
                        res.hour = self._adjust_ampm(res.hour, value)
                        res.ampm = value
                    elif fuzzy:
                        skipped_idxs.append(i)
                elif self._could_be_tzname(res.hour, res.tzname, res.tzoffset, l[i]):
                    res.tzname = l[i]
                    res.tzoffset = info.tzoffset(res.tzname)
                    if i + 1 < len_l and l[i + 1] in ('+', '-'):
                        l[i + 1] = ('+', '-')[l[i + 1] == '+']
                        res.tzoffset = None
                        if info.utczone(res.tzname):
                            res.tzname = None
                elif res.hour is not None and l[i] in ('+', '-'):
                    signal = (-1, 1)[l[i] == '+']
                    len_li = len(l[i + 1])
                    if len_li == 4:
                        hour_offset = int(l[i + 1][:2])
                        min_offset = int(l[i + 1][2:])
                    elif i + 2 < len_l and l[i + 2] == ':':
                        hour_offset = int(l[i + 1])
                        min_offset = int(l[i + 3])
                        i += 2
                    elif len_li <= 2:
                        hour_offset = int(l[i + 1][:2])
                        min_offset = 0
                    else:
                        raise ValueError(timestr)
                    res.tzoffset = signal * (hour_offset * 3600 + min_offset * 60)
                    if i + 5 < len_l and info.jump(l[i + 2]) and (l[i + 3] == '(') and (l[i + 5] == ')') and (3 <= len(l[i + 4])) and self._could_be_tzname(res.hour, res.tzname, None, l[i + 4]):
                        res.tzname = l[i + 4]
                        i += 4
                    i += 1
                elif not (info.jump(l[i]) or fuzzy):
                    raise ValueError(timestr)
                else:
                    skipped_idxs.append(i)
                i += 1
            year, month, day = ymd.resolve_ymd(yearfirst, dayfirst)
            res.century_specified = ymd.century_specified
            res.year = year
            res.month = month
            res.day = day
        except (IndexError, ValueError):
            return (None, None)
        if not info.validate(res):
            return (None, None)
        if fuzzy_with_tokens:
            skipped_tokens = self._recombine_skipped(l, skipped_idxs)
            return (res, tuple(skipped_tokens))
        else:
            return (res, None)

    def _parse_numeric_token(self, tokens, idx, info, ymd, res, fuzzy):
        value_repr = tokens[idx]
        try:
            value = self._to_decimal(value_repr)
        except Exception as e:
            six.raise_from(ValueError('Unknown numeric token'), e)
        len_li = len(value_repr)
        len_l = len(tokens)
        if len(ymd) == 3 and len_li in (2, 4) and (res.hour is None) and (idx + 1 >= len_l or (tokens[idx + 1] != ':' and info.hms(tokens[idx + 1]) is None)):
            s = tokens[idx]
            res.hour = int(s[:2])
            if len_li == 4:
                res.minute = int(s[2:])
        elif len_li == 6 or (len_li > 6 and tokens[idx].find('.') == 6):
            s = tokens[idx]
            if not ymd and '.' not in tokens[idx]:
                ymd.append(s[:2])
                ymd.append(s[2:4])
                ymd.append(s[4:])
            else:
                res.hour = int(s[:2])
                res.minute = int(s[2:4])
                res.second, res.microsecond = self._parsems(s[4:])
        elif len_li in (8, 12, 14):
            s = tokens[idx]
            ymd.append(s[:4], 'Y')
            ymd.append(s[4:6])
            ymd.append(s[6:8])
            if len_li > 8:
                res.hour = int(s[8:10])
                res.minute = int(s[10:12])
                if len_li > 12:
                    res.second = int(s[12:])
        elif self._find_hms_idx(idx, tokens, info, allow_jump=True) is not None:
            hms_idx = self._find_hms_idx(idx, tokens, info, allow_jump=True)
            idx, hms = self._parse_hms(idx, tokens, info, hms_idx)
            if hms is not None:
                self._assign_hms(res, value_repr, hms)
        elif idx + 2 < len_l and tokens[idx + 1] == ':':
            res.hour = int(value)
            value = self._to_decimal(tokens[idx + 2])
            res.minute, res.second = self._parse_min_sec(value)
            if idx + 4 < len_l and tokens[idx + 3] == ':':
                res.second, res.microsecond = self._parsems(tokens[idx + 4])
                idx += 2
            idx += 2
        elif idx + 1 < len_l and tokens[idx + 1] in ('-', '/', '.'):
            sep = tokens[idx + 1]
            ymd.append(value_repr)
            if idx + 2 < len_l and (not info.jump(tokens[idx + 2])):
                if tokens[idx + 2].isdigit():
                    ymd.append(tokens[idx + 2])
                else:
                    value = info.month(tokens[idx + 2])
                    if value is not None:
                        ymd.append(value, 'M')
                    else:
                        raise ValueError()
                if idx + 3 < len_l and tokens[idx + 3] == sep:
                    value = info.month(tokens[idx + 4])
                    if value is not None:
                        ymd.append(value, 'M')
                    else:
                        ymd.append(tokens[idx + 4])
                    idx += 2
                idx += 1
            idx += 1
        elif idx + 1 >= len_l or info.jump(tokens[idx + 1]):
            if idx + 2 < len_l and info.ampm(tokens[idx + 2]) is not None:
                hour = int(value)
                res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 2]))
                idx += 1
            else:
                ymd.append(value)
            idx += 1
        elif info.ampm(tokens[idx + 1]) is not None and 0 <= value < 24:
            hour = int(value)
            res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 1]))
            idx += 1
        elif ymd.could_be_day(value):
            ymd.append(value)
        elif not fuzzy:
            raise ValueError()
        return idx

    def _find_hms_idx(self, idx, tokens, info, allow_jump):
        len_l = len(tokens)
        if idx + 1 < len_l and info.hms(tokens[idx + 1]) is not None:
            hms_idx = idx + 1
        elif allow_jump and idx + 2 < len_l and (tokens[idx + 1] == ' ') and (info.hms(tokens[idx + 2]) is not None):
            hms_idx = idx + 2
        elif idx > 0 and info.hms(tokens[idx - 1]) is not None:
            hms_idx = idx - 1
        elif 1 < idx == len_l - 1 and tokens[idx - 1] == ' ' and (info.hms(tokens[idx - 2]) is not None):
            hms_idx = idx - 2
        else:
            hms_idx = None
        return hms_idx

    def _assign_hms(self, res, value_repr, hms):
        value = self._to_decimal(value_repr)
        if hms == 0:
            res.hour = int(value)
            if value % 1:
                res.minute = int(60 * (value % 1))
        elif hms == 1:
            res.minute, res.second = self._parse_min_sec(value)
        elif hms == 2:
            res.second, res.microsecond = self._parsems(value_repr)

    def _could_be_tzname(self, hour, tzname, tzoffset, token):
        return hour is not None and tzname is None and (tzoffset is None) and (len(token) <= 5) and (all((x in string.ascii_uppercase for x in token)) or token in self.info.UTCZONE)

    def _ampm_valid(self, hour, ampm, fuzzy):
        """
        For fuzzy parsing, 'a' or 'am' (both valid English words)
        may erroneously trigger the AM/PM flag. Deal with that
        here.
        """
        val_is_ampm = True
        if fuzzy and ampm is not None:
            val_is_ampm = False
        if hour is None:
            if fuzzy:
                val_is_ampm = False
            else:
                raise ValueError('No hour specified with AM or PM flag.')
        elif not 0 <= hour <= 12:
            if fuzzy:
                val_is_ampm = False
            else:
                raise ValueError('Invalid hour specified for 12-hour clock.')
        return val_is_ampm

    def _adjust_ampm(self, hour, ampm):
        if hour < 12 and ampm == 1:
            hour += 12
        elif hour == 12 and ampm == 0:
            hour = 0
        return hour

    def _parse_min_sec(self, value):
        minute = int(value)
        second = None
        sec_remainder = value % 1
        if sec_remainder:
            second = int(60 * sec_remainder)
        return (minute, second)

    def _parse_hms(self, idx, tokens, info, hms_idx):
        if hms_idx is None:
            hms = None
            new_idx = idx
        elif hms_idx > idx:
            hms = info.hms(tokens[hms_idx])
            new_idx = hms_idx
        else:
            hms = info.hms(tokens[hms_idx]) + 1
            new_idx = idx
        return (new_idx, hms)

    def _parsems(self, value):
        """Parse a I[.F] seconds value into (seconds, microseconds)."""
        if '.' not in value:
            return (int(value), 0)
        else:
            i, f = value.split('.')
            return (int(i), int(f.ljust(6, '0')[:6]))

    def _to_decimal(self, val):
        try:
            decimal_value = Decimal(val)
            if not decimal_value.is_finite():
                raise ValueError('Converted decimal value is infinite or NaN')
        except Exception as e:
            msg = 'Could not convert %s to decimal' % val
            six.raise_from(ValueError(msg), e)
        else:
            return decimal_value

    def _build_tzinfo(self, tzinfos, tzname, tzoffset):
        if callable(tzinfos):
            tzdata = tzinfos(tzname, tzoffset)
        else:
            tzdata = tzinfos.get(tzname)
        if isinstance(tzdata, datetime.tzinfo) or tzdata is None:
            tzinfo = tzdata
        elif isinstance(tzdata, text_type):
            tzinfo = tz.tzstr(tzdata)
        elif isinstance(tzdata, integer_types):
            tzinfo = tz.tzoffset(tzname, tzdata)
        else:
            raise TypeError('Offset must be tzinfo subclass, tz string, or int offset.')
        return tzinfo

    def _build_tzaware(self, naive, res, tzinfos):
        if callable(tzinfos) or (tzinfos and res.tzname in tzinfos):
            tzinfo = self._build_tzinfo(tzinfos, res.tzname, res.tzoffset)
            aware = naive.replace(tzinfo=tzinfo)
            aware = self._assign_tzname(aware, res.tzname)
        elif res.tzname and res.tzname in time.tzname:
            aware = naive.replace(tzinfo=tz.tzlocal())
            aware = self._assign_tzname(aware, res.tzname)
            if aware.tzname() != res.tzname and res.tzname in self.info.UTCZONE:
                aware = aware.replace(tzinfo=tz.UTC)
        elif res.tzoffset == 0:
            aware = naive.replace(tzinfo=tz.UTC)
        elif res.tzoffset:
            aware = naive.replace(tzinfo=tz.tzoffset(res.tzname, res.tzoffset))
        elif not res.tzname and (not res.tzoffset):
            aware = naive
        elif res.tzname:
            warnings.warn('tzname {tzname} identified but not understood.  Pass `tzinfos` argument in order to correctly return a timezone-aware datetime.  In a future version, this will raise an exception.'.format(tzname=res.tzname), category=UnknownTimezoneWarning)
            aware = naive
        return aware

    def _build_naive(self, res, default):
        repl = {}
        for attr in ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond'):
            value = getattr(res, attr)
            if value is not None:
                repl[attr] = value
        if 'day' not in repl:
            cyear = default.year if res.year is None else res.year
            cmonth = default.month if res.month is None else res.month
            cday = default.day if res.day is None else res.day
            if cday > monthrange(cyear, cmonth)[1]:
                repl['day'] = monthrange(cyear, cmonth)[1]
        naive = default.replace(**repl)
        if res.weekday is not None and (not res.day):
            naive = naive + relativedelta.relativedelta(weekday=res.weekday)
        return naive

    def _assign_tzname(self, dt, tzname):
        if dt.tzname() != tzname:
            new_dt = tz.enfold(dt, fold=1)
            if new_dt.tzname() == tzname:
                return new_dt
        return dt

    def _recombine_skipped(self, tokens, skipped_idxs):
        """
        >>> tokens = ["foo", " ", "bar", " ", "19June2000", "baz"]
        >>> skipped_idxs = [0, 1, 2, 5]
        >>> _recombine_skipped(tokens, skipped_idxs)
        ["foo bar", "baz"]
        """
        skipped_tokens = []
        for i, idx in enumerate(sorted(skipped_idxs)):
            if i > 0 and idx - 1 == skipped_idxs[i - 1]:
                skipped_tokens[-1] = skipped_tokens[-1] + tokens[idx]
            else:
                skipped_tokens.append(tokens[idx])
        return skipped_tokens