import logging
import re
from .compat import string_types
from .util import parse_requirement
def _semantic_key(s):

    def make_tuple(s, absent):
        if s is None:
            result = (absent,)
        else:
            parts = s[1:].split('.')
            result = tuple([p.zfill(8) if p.isdigit() else p for p in parts])
        return result
    m = is_semver(s)
    if not m:
        raise UnsupportedVersionError(s)
    groups = m.groups()
    major, minor, patch = [int(i) for i in groups[:3]]
    pre, build = (make_tuple(groups[3], '|'), make_tuple(groups[5], '*'))
    return ((major, minor, patch), pre, build)