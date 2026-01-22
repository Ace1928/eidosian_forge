from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
from .._utils import to_rgba
from .._utils.registry import RegistryHierarchyMeta
from ..exceptions import PlotnineError, deprecated_themeable_name
from .elements import element_blank
from .elements.element_base import element_base
class line(axis_line, axis_ticks, panel_grid, legend_ticks):
    """
    All line elements

    Parameters
    ----------
    theme_element : element_line
    """

    @property
    def rcParams(self) -> dict[str, Any]:
        rcParams = super().rcParams
        color = self.properties.get('color')
        linewidth = self.properties.get('linewidth')
        linestyle = self.properties.get('linestyle')
        d = {}
        if color:
            d['axes.edgecolor'] = color
            d['xtick.color'] = color
            d['ytick.color'] = color
            d['grid.color'] = color
        if linewidth:
            d['axes.linewidth'] = linewidth
            d['xtick.major.width'] = linewidth
            d['xtick.minor.width'] = linewidth
            d['ytick.major.width'] = linewidth
            d['ytick.minor.width'] = linewidth
            d['grid.linewidth'] = linewidth
        if linestyle:
            d['grid.linestyle'] = linestyle
        rcParams.update(d)
        return rcParams