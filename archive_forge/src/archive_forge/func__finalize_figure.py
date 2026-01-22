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
def _finalize_figure(self, p: Plot) -> None:
    for sub in self._subplots:
        ax = sub['ax']
        for axis in 'xy':
            axis_key = sub[axis]
            axis_obj = getattr(ax, f'{axis}axis')
            if axis_key in p._limits or axis in p._limits:
                convert_units = getattr(ax, f'{axis}axis').convert_units
                a, b = p._limits.get(axis_key) or p._limits[axis]
                lo = a if a is None else convert_units(a)
                hi = b if b is None else convert_units(b)
                if isinstance(a, str):
                    lo = cast(float, lo) - 0.5
                if isinstance(b, str):
                    hi = cast(float, hi) + 0.5
                ax.set(**{f'{axis}lim': (lo, hi)})
            if axis_key in self._scales:
                self._scales[axis_key]._finalize(p, axis_obj)
    if (engine_name := p._layout_spec.get('engine', default)) is not default:
        set_layout_engine(self._figure, engine_name)
    elif p._target is None:
        set_layout_engine(self._figure, 'tight')
    if (extent := p._layout_spec.get('extent')) is not None:
        engine = get_layout_engine(self._figure)
        if engine is None:
            self._figure.subplots_adjust(*extent)
        else:
            left, bottom, right, top = extent
            width, height = (right - left, top - bottom)
            try:
                engine.set(rect=[left, bottom, width, height])
            except TypeError:
                pass