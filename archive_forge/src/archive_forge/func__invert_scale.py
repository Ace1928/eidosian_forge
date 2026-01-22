from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
def _invert_scale(self, ax, data, vars=('x', 'y')):
    """Undo scaling after computation so data are plotted correctly."""
    for var in vars:
        _, inv = _get_transform_functions(ax, var[0])
        if var == self.orient and 'width' in data:
            hw = data['width'] / 2
            data['edge'] = inv(data[var] - hw)
            data['width'] = inv(data[var] + hw) - data['edge'].to_numpy()
        for suf in ['', 'min', 'max']:
            if (col := f'{var}{suf}') in data:
                data[col] = inv(data[col])