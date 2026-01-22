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
def _scatter_legend_artist(**kws):
    kws = normalize_kwargs(kws, mpl.collections.PathCollection)
    edgecolor = kws.pop('edgecolor', None)
    rc = mpl.rcParams
    line_kws = {'linestyle': '', 'marker': kws.pop('marker', 'o'), 'markersize': np.sqrt(kws.pop('s', rc['lines.markersize'] ** 2)), 'markerfacecolor': kws.pop('facecolor', kws.get('color')), 'markeredgewidth': kws.pop('linewidth', 0), **kws}
    if edgecolor is not None:
        if edgecolor == 'face':
            line_kws['markeredgecolor'] = line_kws['markerfacecolor']
        else:
            line_kws['markeredgecolor'] = edgecolor
    return mpl.lines.Line2D([], [], **line_kws)