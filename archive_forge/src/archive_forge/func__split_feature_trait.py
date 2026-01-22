from __future__ import annotations
import logging # isort:skip
import sys
from collections.abc import Iterable
import numpy as np
from ..core.properties import ColorSpec
from ..models import ColumnarDataSource, ColumnDataSource, GlyphRenderer
from ..util.strings import nice_join
from ._legends import pop_legend_kwarg, update_legend
def _split_feature_trait(ft):
    """Feature is up to first '_'. Ex. 'line_color' => ['line', 'color']"""
    ft = ft.split('_', 1)
    return ft if len(ft) == 2 else [*ft, None]