import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
def _default_color(method, hue, color, kws, saturation=1):
    """If needed, get a default color by using the matplotlib property cycle."""
    if hue is not None:
        return None
    kws = kws.copy()
    kws.pop('label', None)
    if color is not None:
        if saturation < 1:
            color = desaturate(color, saturation)
        return color
    elif method.__name__ == 'plot':
        color = normalize_kwargs(kws, mpl.lines.Line2D).get('color')
        scout, = method([], [], scalex=False, scaley=False, color=color)
        color = scout.get_color()
        scout.remove()
    elif method.__name__ == 'scatter':
        scout_size = max((np.atleast_1d(kws.get(key, [])).shape[0] for key in ['s', 'c', 'fc', 'facecolor', 'facecolors']))
        scout_x = scout_y = np.full(scout_size, np.nan)
        scout = method(scout_x, scout_y, **kws)
        facecolors = scout.get_facecolors()
        if not len(facecolors):
            single_color = False
        else:
            single_color = np.unique(facecolors, axis=0).shape[0] == 1
        if 'c' not in kws and single_color:
            color = to_rgb(facecolors[0])
        scout.remove()
    elif method.__name__ == 'bar':
        scout, = method([np.nan], [np.nan], **kws)
        color = to_rgb(scout.get_facecolor())
        scout.remove()
        method.__self__.containers.pop(-1)
    elif method.__name__ == 'fill_between':
        kws = normalize_kwargs(kws, mpl.collections.PolyCollection)
        scout = method([], [], **kws)
        facecolor = scout.get_facecolor()
        color = to_rgb(facecolor[0])
        scout.remove()
    if saturation < 1:
        color = desaturate(color, saturation)
    return color