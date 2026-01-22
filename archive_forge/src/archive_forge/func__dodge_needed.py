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
def _dodge_needed(self):
    """Return True when use of `hue` would cause overlaps."""
    groupers = list({self.orient, 'col', 'row'} & set(self.variables))
    if 'hue' in self.variables:
        orient = self.plot_data[groupers].value_counts()
        paired = self.plot_data[[*groupers, 'hue']].value_counts()
        return orient.size != paired.size
    return False