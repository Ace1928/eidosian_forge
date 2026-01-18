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
def validate_fonttype(s):
    """
    Confirm that this is a Postscript or PDF font type that we know how to
    convert to.
    """
    fonttypes = {'type3': 3, 'truetype': 42}
    try:
        fonttype = validate_int(s)
    except ValueError:
        try:
            return fonttypes[s.lower()]
        except KeyError as e:
            raise ValueError('Supported Postscript/PDF font types are %s' % list(fonttypes)) from e
    else:
        if fonttype not in fonttypes.values():
            raise ValueError('Supported Postscript/PDF font types are %s' % list(fonttypes.values()))
        return fonttype