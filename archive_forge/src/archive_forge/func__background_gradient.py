from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
def _background_gradient(data, cmap: str | Colormap='PuBu', low: float=0, high: float=0, text_color_threshold: float=0.408, vmin: float | None=None, vmax: float | None=None, gmap: Sequence | np.ndarray | DataFrame | Series | None=None, text_only: bool=False):
    """
    Color background in a range according to the data or a gradient map
    """
    if gmap is None:
        gmap = data.to_numpy(dtype=float, na_value=np.nan)
    else:
        gmap = _validate_apply_axis_arg(gmap, 'gmap', float, data)
    with _mpl(Styler.background_gradient) as (_, _matplotlib):
        smin = np.nanmin(gmap) if vmin is None else vmin
        smax = np.nanmax(gmap) if vmax is None else vmax
        rng = smax - smin
        norm = _matplotlib.colors.Normalize(smin - rng * low, smax + rng * high)
        if cmap is None:
            rgbas = _matplotlib.colormaps[_matplotlib.rcParams['image.cmap']](norm(gmap))
        else:
            rgbas = _matplotlib.colormaps.get_cmap(cmap)(norm(gmap))

        def relative_luminance(rgba) -> float:
            """
            Calculate relative luminance of a color.

            The calculation adheres to the W3C standards
            (https://www.w3.org/WAI/GL/wiki/Relative_luminance)

            Parameters
            ----------
            color : rgb or rgba tuple

            Returns
            -------
            float
                The relative luminance as a value from 0 to 1
            """
            r, g, b = (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgba[:3])
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def css(rgba, text_only) -> str:
            if not text_only:
                dark = relative_luminance(rgba) < text_color_threshold
                text_color = '#f1f1f1' if dark else '#000000'
                return f'background-color: {_matplotlib.colors.rgb2hex(rgba)};color: {text_color};'
            else:
                return f'color: {_matplotlib.colors.rgb2hex(rgba)};'
        if data.ndim == 1:
            return [css(rgba, text_only) for rgba in rgbas]
        else:
            return DataFrame([[css(rgba, text_only) for rgba in row] for row in rgbas], index=data.index, columns=data.columns)