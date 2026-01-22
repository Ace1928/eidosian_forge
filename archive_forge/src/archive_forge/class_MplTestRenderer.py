from __future__ import annotations
from collections.abc import Sequence
import io
from typing import TYPE_CHECKING, Any, cast
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
from contourpy import FillType, LineType
from contourpy.convert import convert_filled, convert_lines
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.util.mpl_util import filled_to_mpl_paths, lines_to_mpl_paths
from contourpy.util.renderer import Renderer
class MplTestRenderer(MplRenderer):
    """Test renderer implemented using Matplotlib.

    No whitespace around plots and no spines/ticks displayed.
    Uses Agg backend, so can only save to file/buffer, cannot call ``show()``.
    """

    def __init__(self, nrows: int=1, ncols: int=1, figsize: tuple[float, float]=(9, 9)) -> None:
        gridspec = {'left': 0.01, 'right': 0.99, 'top': 0.99, 'bottom': 0.01, 'wspace': 0.01, 'hspace': 0.01}
        super().__init__(nrows, ncols, figsize, show_frame=True, backend='Agg', gridspec_kw=gridspec)
        for ax in self._axes:
            ax.set_xmargin(0.0)
            ax.set_ymargin(0.0)
            ax.set_xticks([])
            ax.set_yticks([])
        self._want_tight = False