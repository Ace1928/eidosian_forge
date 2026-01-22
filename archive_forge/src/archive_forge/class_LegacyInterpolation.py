from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
class LegacyInterpolation(Interpolation):
    """Deprecated interpolation used in old versions of ConfigParser.
    Use BasicInterpolation or ExtendedInterpolation instead."""
    _KEYCRE = re.compile('%\\(([^)]*)\\)s|.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn('LegacyInterpolation has been deprecated since Python 3.2 and will be removed from the configparser module in Python 3.13. Use BasicInterpolation or ExtendedInterpolation instead.', DeprecationWarning, stacklevel=2)

    def before_get(self, parser, section, option, value, vars):
        rawval = value
        depth = MAX_INTERPOLATION_DEPTH
        while depth:
            depth -= 1
            if value and '%(' in value:
                replace = functools.partial(self._interpolation_replace, parser=parser)
                value = self._KEYCRE.sub(replace, value)
                try:
                    value = value % vars
                except KeyError as e:
                    raise InterpolationMissingOptionError(option, section, rawval, e.args[0]) from None
            else:
                break
        if value and '%(' in value:
            raise InterpolationDepthError(option, section, rawval)
        return value

    def before_set(self, parser, section, option, value):
        return value

    @staticmethod
    def _interpolation_replace(match, parser):
        s = match.group(1)
        if s is None:
            return match.group()
        else:
            return '%%(%s)s' % parser.optionxform(s)