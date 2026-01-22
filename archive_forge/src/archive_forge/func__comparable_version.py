import collections
import os
import re
import sys
import functools
import itertools
def _comparable_version(version):
    result = []
    for v in _component_re.split(version):
        if v not in '._+-':
            try:
                v = int(v, 10)
                t = 100
            except ValueError:
                t = _ver_stages.get(v, 0)
            result.extend((t, v))
    return result