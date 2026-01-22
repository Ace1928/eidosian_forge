from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _who_when(self, s, cmd, section, accept_just_who=False):
    """Parse who and when information from a string.

        :return: a tuple of (name,email,timestamp,timezone). name may be
            the empty string if only an email address was given.
        """
    match = _WHO_AND_WHEN_RE.search(s)
    if match:
        datestr = match.group(3).lstrip()
        if self.date_parser is None:
            if len(datestr.split(b' ')) == 2:
                date_format = 'raw'
            elif datestr == b'now':
                date_format = 'now'
            else:
                date_format = 'rfc2822'
            self.date_parser = dates.DATE_PARSERS_BY_NAME[date_format]
        try:
            when = self.date_parser(datestr, self.lineno)
        except ValueError:
            print("failed to parse datestr '%s'" % (datestr,))
            raise
        name = match.group(1).rstrip()
        email = match.group(2)
    else:
        match = _WHO_RE.search(s)
        if accept_just_who and match:
            when = dates.DATE_PARSERS_BY_NAME['now']('now')
            name = match.group(1)
            email = match.group(2)
        elif self.strict:
            self.abort(errors.BadFormat, cmd, section, s)
        else:
            name = s
            email = None
            when = dates.DATE_PARSERS_BY_NAME['now']('now')
    if len(name) > 0:
        if name.endswith(b' '):
            name = name[:-1]
    if self.user_mapper:
        name, email = self.user_mapper.map_name_and_email(name, email)
    return Authorship(name, email, when[0], when[1])