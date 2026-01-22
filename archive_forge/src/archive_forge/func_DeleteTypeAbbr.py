from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import six
def DeleteTypeAbbr(suffix, type_abbr='B'):
    """Returns suffix with trailing type abbreviation deleted."""
    if not suffix:
        return suffix
    s = suffix.upper()
    i = len(s)
    for c in reversed(type_abbr.upper()):
        if not i:
            break
        if s[i - 1] == c:
            i -= 1
    return suffix[:i]