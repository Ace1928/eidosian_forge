import logging
import re
from .compat import string_types
from .util import parse_requirement
def _match_ge(self, version, constraint, prefix):
    version, constraint = self._adjust_local(version, constraint, prefix)
    return version >= constraint