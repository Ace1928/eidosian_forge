import time
import locale
import calendar
from re import compile as re_compile
from re import IGNORECASE
from re import escape as re_escape
from datetime import (date as datetime_date,
from _thread import allocate_lock as _thread_allocate_lock
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