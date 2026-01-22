from __future__ import print_function, absolute_import
import sys
import re
import warnings
import types
import keyword
import functools
from shibokensupport.signature.mapping import (type_map, update_mapping,
from shibokensupport.signature.lib.tool import (SimpleNamespace,
from inspect import currentframe
def _parse_arglist(argstr):
    key = '_parse_arglist'
    if key not in _cache:
        regex = build_brace_pattern(level=3, separators=',')
        _cache[key] = re.compile(regex, flags=re.VERBOSE)
    split = _cache[key].split
    return [x.strip() for x in split(argstr) if x.strip() not in ('', ',')]