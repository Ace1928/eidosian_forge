from __future__ import annotations
import typing
from collections.abc import Sequence
from copy import copy, deepcopy
from io import BytesIO
from itertools import chain
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, Iterable, Optional
from warnings import warn
from ._utils import (
from ._utils.context import plot_context
from ._utils.ipython import (
from .coords import coord_cartesian
from .exceptions import PlotnineError, PlotnineWarning
from .facets import facet_null
from .facets.layout import Layout
from .geoms.geom_blank import geom_blank
from .guides.guides import guides
from .iapi import mpl_save_view
from .layer import Layers
from .mapping.aes import aes, make_labels
from .options import get_option
from .scales.scales import Scales
from .themes.theme import theme, theme_get
def _draw_breaks_and_labels(self):
    """
        Draw breaks and labels
        """
    self.facet.strips.draw()
    for layout_info in self.layout.get_details():
        pidx = layout_info.panel_index
        ax = self.axs[pidx]
        panel_params = self.layout.panel_params[pidx]
        self.facet.set_limits_breaks_and_labels(panel_params, ax)
        if not layout_info.axis_x:
            ax.xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
        if not layout_info.axis_y:
            ax.yaxis.set_tick_params(which='both', left=False, labelleft=False)
        if layout_info.axis_x:
            ax.xaxis.set_tick_params(which='both', bottom=True)
        if layout_info.axis_y:
            ax.yaxis.set_tick_params(which='both', left=True)