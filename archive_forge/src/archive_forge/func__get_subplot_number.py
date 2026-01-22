import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def _get_subplot_number(subplot_val):
    """
    Extract the subplot number from a subplot value string.

    'x3' -> 3
    'polar2' -> 2
    'scene' -> 1
    'y' -> 1

    Note: the absence of a subplot number (e.g. 'y') is treated by plotly as
    a subplot number of 1

    Parameters
    ----------
    subplot_val: str
        Subplot string value (e.g. 'scene4')

    Returns
    -------
    int
    """
    match = _subplot_re.match(subplot_val)
    if match:
        subplot_number = int(match.group(1))
    else:
        subplot_number = 1
    return subplot_number