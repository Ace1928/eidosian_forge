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
def validate_whiskers(s):
    try:
        return _listify_validator(validate_float, n=2)(s)
    except (TypeError, ValueError):
        try:
            return float(s)
        except ValueError as e:
            raise ValueError('Not a valid whisker value [float, (float, float)]') from e