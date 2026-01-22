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
def _draw_panel_borders(self):
    """
        Draw Panel boders
        """
    if self.theme.T.is_blank('panel_border'):
        return
    from matplotlib.patches import Rectangle
    for ax in self.axs:
        rect = Rectangle((0, 0), 1, 1, facecolor='none', transform=ax.transAxes, clip_path=ax.patch, clip_on=False)
        self.figure.add_artist(rect)
        self.theme.targets.panel_border.append(rect)