from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def _get_conv(self, section, option, conv, *, raw=False, vars=None, fallback=_UNSET, **kwargs):
    try:
        return self._get(section, conv, option, raw=raw, vars=vars, **kwargs)
    except (NoSectionError, NoOptionError):
        if fallback is _UNSET:
            raise
        return fallback