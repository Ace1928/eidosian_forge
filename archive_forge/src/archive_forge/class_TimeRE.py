import time
import locale
import calendar
from re import compile as re_compile
from re import IGNORECASE
from re import escape as re_escape
from datetime import (date as datetime_date,
from _thread import allocate_lock as _thread_allocate_lock
class TimeRE(dict):
    """Handle conversion from format directives to regexes."""

    def __init__(self, locale_time=None):
        """Create keys/values.

        Order of execution is important for dependency reasons.

        """
        if locale_time:
            self.locale_time = locale_time
        else:
            self.locale_time = LocaleTime()
        base = super()
        base.__init__({'d': '(?P<d>3[0-1]|[1-2]\\d|0[1-9]|[1-9]| [1-9])', 'f': '(?P<f>[0-9]{1,6})', 'H': '(?P<H>2[0-3]|[0-1]\\d|\\d)', 'I': '(?P<I>1[0-2]|0[1-9]|[1-9])', 'G': '(?P<G>\\d\\d\\d\\d)', 'j': '(?P<j>36[0-6]|3[0-5]\\d|[1-2]\\d\\d|0[1-9]\\d|00[1-9]|[1-9]\\d|0[1-9]|[1-9])', 'm': '(?P<m>1[0-2]|0[1-9]|[1-9])', 'M': '(?P<M>[0-5]\\d|\\d)', 'S': '(?P<S>6[0-1]|[0-5]\\d|\\d)', 'U': '(?P<U>5[0-3]|[0-4]\\d|\\d)', 'w': '(?P<w>[0-6])', 'u': '(?P<u>[1-7])', 'V': '(?P<V>5[0-3]|0[1-9]|[1-4]\\d|\\d)', 'y': '(?P<y>\\d\\d)', 'Y': '(?P<Y>\\d\\d\\d\\d)', 'z': '(?P<z>[+-]\\d\\d:?[0-5]\\d(:?[0-5]\\d(\\.\\d{1,6})?)?|(?-i:Z))', 'A': self.__seqToRE(self.locale_time.f_weekday, 'A'), 'a': self.__seqToRE(self.locale_time.a_weekday, 'a'), 'B': self.__seqToRE(self.locale_time.f_month[1:], 'B'), 'b': self.__seqToRE(self.locale_time.a_month[1:], 'b'), 'p': self.__seqToRE(self.locale_time.am_pm, 'p'), 'Z': self.__seqToRE((tz for tz_names in self.locale_time.timezone for tz in tz_names), 'Z'), '%': '%'})
        base.__setitem__('W', base.__getitem__('U').replace('U', 'W'))
        base.__setitem__('c', self.pattern(self.locale_time.LC_date_time))
        base.__setitem__('x', self.pattern(self.locale_time.LC_date))
        base.__setitem__('X', self.pattern(self.locale_time.LC_time))

    def __seqToRE(self, to_convert, directive):
        """Convert a list to a regex string for matching a directive.

        Want possible matching values to be from longest to shortest.  This
        prevents the possibility of a match occurring for a value that also
        a substring of a larger value that should have matched (e.g., 'abc'
        matching when 'abcdef' should have been the match).

        """
        to_convert = sorted(to_convert, key=len, reverse=True)
        for value in to_convert:
            if value != '':
                break
        else:
            return ''
        regex = '|'.join((re_escape(stuff) for stuff in to_convert))
        regex = '(?P<%s>%s' % (directive, regex)
        return '%s)' % regex

    def pattern(self, format):
        """Return regex pattern for the format string.

        Need to make sure that any characters that might be interpreted as
        regex syntax are escaped.

        """
        processed_format = ''
        regex_chars = re_compile('([\\\\.^$*+?\\(\\){}\\[\\]|])')
        format = regex_chars.sub('\\\\\\1', format)
        whitespace_replacement = re_compile('\\s+')
        format = whitespace_replacement.sub('\\\\s+', format)
        while '%' in format:
            directive_index = format.index('%') + 1
            processed_format = '%s%s%s' % (processed_format, format[:directive_index - 1], self[format[directive_index]])
            format = format[directive_index + 1:]
        return '%s%s' % (processed_format, format)

    def compile(self, format):
        """Return a compiled re object for the format string."""
        return re_compile(self.pattern(format), IGNORECASE)