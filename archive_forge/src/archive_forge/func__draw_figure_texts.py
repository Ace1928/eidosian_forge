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
def _draw_figure_texts(self):
    """
        Draw title, x label, y label and caption onto the figure
        """
    figure = self.figure
    theme = self.theme
    targets = theme.targets
    title = self.labels.get('title', '')
    subtitle = self.labels.get('subtitle', '')
    caption = self.labels.get('caption', '')
    labels = self.coordinates.labels(self.layout.set_xy_labels(self.labels))
    if title:
        targets.plot_title = figure.text(0, 0, title)
    if subtitle:
        targets.plot_subtitle = figure.text(0, 0, subtitle)
    if caption:
        targets.plot_caption = figure.text(0, 0, caption)
    if labels.x:
        targets.axis_title_x = figure.text(0, 0, labels.x)
    if labels.y:
        targets.axis_title_y = figure.text(0, 0, labels.y)