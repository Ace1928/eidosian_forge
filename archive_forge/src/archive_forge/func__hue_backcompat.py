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
def _hue_backcompat(self, color, palette, hue_order, force_hue=False):
    """Implement backwards compatibility for hue parametrization.

        Note: the force_hue parameter is used so that functions can be shown to
        pass existing tests during refactoring and then tested for new behavior.
        It can be removed after completion of the work.

        """
    default_behavior = color is None or palette is not None
    if force_hue and 'hue' not in self.variables and default_behavior:
        self._redundant_hue = True
        self.plot_data['hue'] = self.plot_data[self.orient]
        self.variables['hue'] = self.variables[self.orient]
        self.var_types['hue'] = 'categorical'
        hue_order = self.var_levels[self.orient]
        if isinstance(palette, dict):
            palette = {str(k): v for k, v in palette.items()}
    else:
        if 'hue' in self.variables:
            redundant = (self.plot_data['hue'] == self.plot_data[self.orient]).all()
        else:
            redundant = False
        self._redundant_hue = redundant
    if 'hue' in self.variables and palette is None and (color is not None):
        if not isinstance(color, str):
            color = mpl.colors.to_hex(color)
        palette = f'dark:{color}'
        msg = f"\n\nSetting a gradient palette using color= is deprecated and will be removed in v0.14.0. Set `palette='{palette}'` for the same effect.\n"
        warnings.warn(msg, FutureWarning, stacklevel=3)
    return (palette, hue_order)