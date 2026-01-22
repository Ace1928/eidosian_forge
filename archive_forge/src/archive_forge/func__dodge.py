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
def _dodge(self, keys, data):
    """Apply a dodge transform to coordinates in place."""
    if 'hue' not in self.variables:
        return
    hue_idx = self._hue_map.levels.index(keys['hue'])
    n = len(self._hue_map.levels)
    data['width'] /= n
    full_width = data['width'] * n
    offset = data['width'] * hue_idx + data['width'] / 2 - full_width / 2
    data[self.orient] += offset