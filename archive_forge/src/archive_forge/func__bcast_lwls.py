import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
@staticmethod
def _bcast_lwls(linewidths, dashes):
    """
        Internal helper function to broadcast + scale ls/lw

        In the collection drawing code, the linewidth and linestyle are cycled
        through as circular buffers (via ``v[i % len(v)]``).  Thus, if we are
        going to scale the dash pattern at set time (not draw time) we need to
        do the broadcasting now and expand both lists to be the same length.

        Parameters
        ----------
        linewidths : list
            line widths of collection
        dashes : list
            dash specification (offset, (dash pattern tuple))

        Returns
        -------
        linewidths, dashes : list
            Will be the same length, dashes are scaled by paired linewidth
        """
    if mpl.rcParams['_internal.classic_mode']:
        return (linewidths, dashes)
    if len(dashes) != len(linewidths):
        l_dashes = len(dashes)
        l_lw = len(linewidths)
        gcd = math.gcd(l_dashes, l_lw)
        dashes = list(dashes) * (l_lw // gcd)
        linewidths = list(linewidths) * (l_dashes // gcd)
    dashes = [mlines._scale_dashes(o, d, lw) for (o, d), lw in zip(dashes, linewidths)]
    return (linewidths, dashes)