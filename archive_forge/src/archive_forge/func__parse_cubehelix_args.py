import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def _parse_cubehelix_args(argstr):
    """Turn stringified cubehelix params into args/kwargs."""
    if argstr.startswith('ch:'):
        argstr = argstr[3:]
    if argstr.endswith('_r'):
        reverse = True
        argstr = argstr[:-2]
    else:
        reverse = False
    if not argstr:
        return ([], {'reverse': reverse})
    all_args = argstr.split(',')
    args = [float(a.strip(' ')) for a in all_args if '=' not in a]
    kwargs = [a.split('=') for a in all_args if '=' in a]
    kwargs = {k.strip(' '): float(v.strip(' ')) for k, v in kwargs}
    kwarg_map = dict(s='start', r='rot', g='gamma', h='hue', l='light', d='dark')
    kwargs = {kwarg_map.get(k, k): v for k, v in kwargs.items()}
    if reverse:
        kwargs['reverse'] = True
    return (args, kwargs)