from __future__ import annotations
import io
import os
import re
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from collections import abc
from collections.abc import Callable, Generator
from typing import Any, List, Literal, Optional, cast
from xml.etree import ElementTree
from cycler import cycler
import pandas as pd
from pandas import DataFrame, Series, Index
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.typing import (
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.rules import categorical_order
from seaborn._compat import get_layout_engine, set_layout_engine
from seaborn.utils import _version_predates
from seaborn.rcmod import axes_style, plotting_context
from seaborn.palettes import color_palette
from typing import TYPE_CHECKING, TypedDict
def _setup_scales(self, p: Plot, common: PlotData, layers: list[Layer], variables: list[str] | None=None) -> None:
    if variables is None:
        variables = []
        for layer in layers:
            variables.extend(layer['data'].frame.columns)
            for df in layer['data'].frames.values():
                variables.extend((str(v) for v in df if v not in variables))
        variables = [v for v in variables if v not in self._scales]
    for var in variables:
        m = re.match('^(?P<coord>(?P<axis>x|y)\\d*).*', var)
        if m is None:
            coord = axis = None
        else:
            coord = m['coord']
            axis = m['axis']
        prop_key = var if axis is None else axis
        scale_key = var if coord is None else coord
        if prop_key not in PROPERTIES:
            continue
        cols = [var, 'col', 'row']
        parts = [common.frame.filter(cols)]
        for layer in layers:
            parts.append(layer['data'].frame.filter(cols))
            for df in layer['data'].frames.values():
                parts.append(df.filter(cols))
        var_df = pd.concat(parts, ignore_index=True)
        prop = PROPERTIES[prop_key]
        scale = self._get_scale(p, scale_key, prop, var_df[var])
        if scale_key not in p._variables:
            scale._priority = 0
        if axis is None:
            share_state = None
            subplots = []
        else:
            share_state = self._subplots.subplot_spec[f'share{axis}']
            subplots = [view for view in self._subplots if view[axis] == coord]
        if scale is None:
            self._scales[var] = Scale._identity()
        else:
            try:
                self._scales[var] = scale._setup(var_df[var], prop)
            except Exception as err:
                raise PlotSpecError._during('Scale setup', var) from err
        if axis is None or (var != coord and coord in p._variables):
            continue
        transformed_data = []
        for layer in layers:
            index = layer['data'].frame.index
            empty_series = pd.Series(dtype=float, index=index, name=var)
            transformed_data.append(empty_series)
        for view in subplots:
            axis_obj = getattr(view['ax'], f'{axis}axis')
            seed_values = self._get_subplot_data(var_df, var, view, share_state)
            view_scale = scale._setup(seed_values, prop, axis=axis_obj)
            view['ax'].set(**{f'{axis}scale': view_scale._matplotlib_scale})
            for layer, new_series in zip(layers, transformed_data):
                layer_df = layer['data'].frame
                if var not in layer_df:
                    continue
                idx = self._get_subplot_index(layer_df, view)
                try:
                    new_series.loc[idx] = view_scale(layer_df.loc[idx, var])
                except Exception as err:
                    spec_error = PlotSpecError._during('Scaling operation', var)
                    raise spec_error from err
        for layer, new_series in zip(layers, transformed_data):
            layer_df = layer['data'].frame
            if var in layer_df:
                layer_df[var] = pd.to_numeric(new_series)