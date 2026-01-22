import ast
from functools import lru_cache, reduce
from numbers import Real
import operator
import os
import re
import numpy as np
from matplotlib import _api, cbook
from matplotlib.cbook import ls_mapper
from matplotlib.colors import Colormap, is_color_like
from matplotlib._fontconfig_pattern import parse_fontconfig_pattern
from matplotlib._enums import JoinStyle, CapStyle
from cycler import Cycler, cycler as ccycler
class ValidateInStrings:

    def __init__(self, key, valid, ignorecase=False, *, _deprecated_since=None):
        """*valid* is a list of legal strings."""
        self.key = key
        self.ignorecase = ignorecase
        self._deprecated_since = _deprecated_since

        def func(s):
            if ignorecase:
                return s.lower()
            else:
                return s
        self.valid = {func(k): k for k in valid}

    def __call__(self, s):
        if self._deprecated_since:
            name, = (k for k, v in globals().items() if v is self)
            _api.warn_deprecated(self._deprecated_since, name=name, obj_type='function')
        if self.ignorecase and isinstance(s, str):
            s = s.lower()
        if s in self.valid:
            return self.valid[s]
        msg = f'{s!r} is not a valid value for {self.key}; supported values are {[*self.valid.values()]}'
        if isinstance(s, str) and (s.startswith('"') and s.endswith('"') or (s.startswith("'") and s.endswith("'"))) and (s[1:-1] in self.valid):
            msg += '; remove quotes surrounding your string'
        raise ValueError(msg)