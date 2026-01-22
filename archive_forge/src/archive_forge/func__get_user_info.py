from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _get_user_info(self, cmd, section, required=True, accept_just_who=False):
    """Parse a user section."""
    line = self.next_line()
    if line.startswith(section + b' '):
        return self._who_when(line[len(section + b' '):], cmd, section, accept_just_who=accept_just_who)
    elif required:
        self.abort(errors.MissingSection, cmd, section)
    else:
        self.push_line(line)
        return None