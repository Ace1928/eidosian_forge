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
def _generate_pairings(self, data: PlotData, pair_variables: dict) -> Generator[tuple[list[dict], DataFrame, dict[str, Scale]], None, None]:
    iter_axes = itertools.product(*[pair_variables.get(axis, [axis]) for axis in 'xy'])
    for x, y in iter_axes:
        subplots = []
        for view in self._subplots:
            if view['x'] == x and view['y'] == y:
                subplots.append(view)
        if data.frame.empty and data.frames:
            out_df = data.frames[x, y].copy()
        elif not pair_variables:
            out_df = data.frame.copy()
        elif data.frame.empty and data.frames:
            out_df = data.frames[x, y].copy()
        else:
            out_df = data.frame.copy()
        scales = self._scales.copy()
        if x in out_df:
            scales['x'] = self._scales[x]
        if y in out_df:
            scales['y'] = self._scales[y]
        for axis, var in zip('xy', (x, y)):
            if axis != var:
                out_df = out_df.rename(columns={var: axis})
                cols = [col for col in out_df if re.match(f'{axis}\\d+', str(col))]
                out_df = out_df.drop(cols, axis=1)
        yield (subplots, out_df, scales)