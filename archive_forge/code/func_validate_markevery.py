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
def validate_markevery(s):
    """
    Validate the markevery property of a Line2D object.

    Parameters
    ----------
    s : None, int, (int, int), slice, float, (float, float), or list[int]

    Returns
    -------
    None, int, (int, int), slice, float, (float, float), or list[int]
    """
    if isinstance(s, (slice, float, int, type(None))):
        return s
    if isinstance(s, tuple):
        if len(s) == 2 and (all((isinstance(e, int) for e in s)) or all((isinstance(e, float) for e in s))):
            return s
        else:
            raise TypeError("'markevery' tuple must be pair of ints or of floats")
    if isinstance(s, list):
        if all((isinstance(e, int) for e in s)):
            return s
        else:
            raise TypeError("'markevery' list must have all elements of type int")
    raise TypeError("'markevery' is of an invalid type")