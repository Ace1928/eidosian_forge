from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
def _adjust_legend_subtitles(legend):
    """Make invisible-handle "subtitles" entries look more like titles."""
    import matplotlib.pyplot as plt
    font_size = plt.rcParams.get('legend.title_fontsize', None)
    hpackers = legend.findobj(plt.matplotlib.offsetbox.VPacker)[0].get_children()
    hpackers = [v for v in hpackers if isinstance(v, plt.matplotlib.offsetbox.HPacker)]
    for hpack in hpackers:
        areas = hpack.get_children()
        if len(areas) < 2:
            continue
        draw_area, text_area = areas
        handles = draw_area.get_children()
        if not all((artist.get_visible() for artist in handles)):
            draw_area.set_width(0)
            for text in text_area.get_children():
                if font_size is not None:
                    text.set_size(font_size)