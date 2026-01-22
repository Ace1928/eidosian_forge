import logging
import re
from .compat import string_types
from .util import parse_requirement
def _adjust_local(self, version, constraint, prefix):
    if prefix:
        strip_local = '+' not in constraint and version._parts[-1]
    else:
        strip_local = not constraint._parts[-1] and version._parts[-1]
    if strip_local:
        s = version._string.split('+', 1)[0]
        version = self.version_class(s)
    return (version, constraint)