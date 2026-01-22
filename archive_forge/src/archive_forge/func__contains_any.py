from __future__ import (absolute_import, division, print_function)
import re
def _contains_any(src, values):
    return any((x in src for x in values))