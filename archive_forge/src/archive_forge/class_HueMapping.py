from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
class HueMapping(SemanticMapping):
    """Mapping that sets artist colors according to data values."""
    palette = None
    norm = None
    cmap = None

    def __init__(self, plotter, palette=None, order=None, norm=None, saturation=1):
        """Map the levels of the `hue` variable to distinct colors.

        Parameters
        ----------
        # TODO add generic parameters

        """
        super().__init__(plotter)
        data = plotter.plot_data.get('hue', pd.Series(dtype=float))
        if isinstance(palette, np.ndarray):
            msg = 'Numpy array is not a supported type for `palette`. Please convert your palette to a list. This will become an error in v0.14'
            warnings.warn(msg, stacklevel=4)
            palette = palette.tolist()
        if data.isna().all():
            if palette is not None:
                msg = 'Ignoring `palette` because no `hue` variable has been assigned.'
                warnings.warn(msg, stacklevel=4)
        else:
            map_type = self.infer_map_type(palette, norm, plotter.input_format, plotter.var_types['hue'])
            if map_type == 'numeric':
                data = pd.to_numeric(data)
                levels, lookup_table, norm, cmap = self.numeric_mapping(data, palette, norm)
            elif map_type == 'categorical':
                cmap = norm = None
                levels, lookup_table = self.categorical_mapping(data, palette, order)
            else:
                cmap = norm = None
                levels, lookup_table = self.categorical_mapping(list(data), palette, order)
            self.saturation = saturation
            self.map_type = map_type
            self.lookup_table = lookup_table
            self.palette = palette
            self.levels = levels
            self.norm = norm
            self.cmap = cmap

    def _lookup_single(self, key):
        """Get the color for a single value, using colormap to interpolate."""
        try:
            value = self.lookup_table[key]
        except KeyError:
            if self.norm is None:
                return (0, 0, 0, 0)
            try:
                normed = self.norm(key)
            except TypeError as err:
                if np.isnan(key):
                    value = (0, 0, 0, 0)
                else:
                    raise err
            else:
                if np.ma.is_masked(normed):
                    normed = np.nan
                value = self.cmap(normed)
        if self.saturation < 1:
            value = desaturate(value, self.saturation)
        return value

    def infer_map_type(self, palette, norm, input_format, var_type):
        """Determine how to implement the mapping."""
        if palette in QUAL_PALETTES:
            map_type = 'categorical'
        elif norm is not None:
            map_type = 'numeric'
        elif isinstance(palette, (dict, list)):
            map_type = 'categorical'
        elif input_format == 'wide':
            map_type = 'categorical'
        else:
            map_type = var_type
        return map_type

    def categorical_mapping(self, data, palette, order):
        """Determine colors when the hue mapping is categorical."""
        levels = categorical_order(data, order)
        n_colors = len(levels)
        if isinstance(palette, dict):
            missing = set(levels) - set(palette)
            if any(missing):
                err = 'The palette dictionary is missing keys: {}'
                raise ValueError(err.format(missing))
            lookup_table = palette
        else:
            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette('husl', n_colors)
            elif isinstance(palette, list):
                colors = self._check_list_length(levels, palette, 'palette')
            else:
                colors = color_palette(palette, n_colors)
            lookup_table = dict(zip(levels, colors))
        return (levels, lookup_table)

    def numeric_mapping(self, data, palette, norm):
        """Determine colors when the hue variable is quantitative."""
        if isinstance(palette, dict):
            levels = list(sorted(palette))
            colors = [palette[k] for k in sorted(palette)]
            cmap = mpl.colors.ListedColormap(colors)
            lookup_table = palette.copy()
        else:
            levels = list(np.sort(remove_na(data.unique())))
            palette = 'ch:' if palette is None else palette
            if isinstance(palette, mpl.colors.Colormap):
                cmap = palette
            else:
                cmap = color_palette(palette, as_cmap=True)
            if norm is None:
                norm = mpl.colors.Normalize()
            elif isinstance(norm, tuple):
                norm = mpl.colors.Normalize(*norm)
            elif not isinstance(norm, mpl.colors.Normalize):
                err = '``hue_norm`` must be None, tuple, or Normalize object.'
                raise ValueError(err)
            if not norm.scaled():
                norm(np.asarray(data.dropna()))
            lookup_table = dict(zip(levels, cmap(norm(levels))))
        return (levels, lookup_table, norm, cmap)