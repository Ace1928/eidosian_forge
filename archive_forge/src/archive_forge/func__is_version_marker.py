import os
import re
import sys
import platform
from .compat import string_types
from .util import in_venv, parse_marker
from .version import LegacyVersion as LV
def _is_version_marker(s):
    return isinstance(s, string_types) and s in _VERSION_MARKERS