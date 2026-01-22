from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import (
def _adjust_fig_for_guide(self, guide) -> None:
    if hasattr(self.fig.canvas, 'get_renderer'):
        renderer = self.fig.canvas.get_renderer()
    else:
        raise RuntimeError('MPL backend has no renderer')
    self.fig.draw(renderer)
    guide_width = guide.get_window_extent(renderer).width / self.fig.dpi
    figure_width = self.fig.get_figwidth()
    total_width = figure_width + guide_width
    self.fig.set_figwidth(total_width)
    self.fig.draw(renderer)
    guide_width = guide.get_window_extent(renderer).width / self.fig.dpi
    space_needed = guide_width / total_width + 0.02
    right = 1 - space_needed
    self.fig.subplots_adjust(right=right)