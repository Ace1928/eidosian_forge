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
def _setup_figure(self, p: Plot, common: PlotData, layers: list[Layer]) -> None:
    subplot_spec = p._subplot_spec.copy()
    facet_spec = p._facet_spec.copy()
    pair_spec = p._pair_spec.copy()
    for axis in 'xy':
        if axis in p._shares:
            subplot_spec[f'share{axis}'] = p._shares[axis]
    for dim in ['col', 'row']:
        if dim in common.frame and dim not in facet_spec['structure']:
            order = categorical_order(common.frame[dim])
            facet_spec['structure'][dim] = order
    self._subplots = subplots = Subplots(subplot_spec, facet_spec, pair_spec)
    self._figure = subplots.init_figure(pair_spec, self._pyplot, p._figure_spec, p._target)
    for sub in subplots:
        ax = sub['ax']
        for axis in 'xy':
            axis_key = sub[axis]
            names = [common.names.get(axis_key), *(layer['data'].names.get(axis_key) for layer in layers)]
            auto_label = next((name for name in names if name is not None), None)
            label = self._resolve_label(p, axis_key, auto_label)
            ax.set(**{f'{axis}label': label})
            axis_obj = getattr(ax, f'{axis}axis')
            visible_side = {'x': 'bottom', 'y': 'left'}.get(axis)
            show_axis_label = sub[visible_side] or not p._pair_spec.get('cross', True) or (axis in p._pair_spec.get('structure', {}) and bool(p._pair_spec.get('wrap')))
            axis_obj.get_label().set_visible(show_axis_label)
            show_tick_labels = show_axis_label or subplot_spec.get(f'share{axis}') not in (True, 'all', {'x': 'col', 'y': 'row'}[axis])
            for group in ('major', 'minor'):
                side = {'x': 'bottom', 'y': 'left'}[axis]
                axis_obj.set_tick_params(**{f'label{side}': show_tick_labels})
                for t in getattr(axis_obj, f'get_{group}ticklabels')():
                    t.set_visible(show_tick_labels)
        title_parts = []
        for dim in ['col', 'row']:
            if sub[dim] is not None:
                val = self._resolve_label(p, 'title', f'{sub[dim]}')
                if dim in p._labels:
                    key = self._resolve_label(p, dim, common.names.get(dim))
                    val = f'{key} {val}'
                title_parts.append(val)
        has_col = sub['col'] is not None
        has_row = sub['row'] is not None
        show_title = has_col and has_row or ((has_col or has_row) and p._facet_spec.get('wrap')) or (has_col and sub['top']) or has_row
        if title_parts:
            title = ' | '.join(title_parts)
            title_text = ax.set_title(title)
            title_text.set_visible(show_title)
        elif not (has_col or has_row):
            title = self._resolve_label(p, 'title', None)
            title_text = ax.set_title(title)