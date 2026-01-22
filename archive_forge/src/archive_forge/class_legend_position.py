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
class legend_position(legend_position_inside):
    """
    Location of legend

    Parameters
    ----------
    theme_element : Literal["right", "left", "top", "bottom", "inside"] |                     tuple[float, float] | Literal["none"]
        Where to put the legend. Along the edge or inside the panels.

        If "inside", the default location is
        [](:class:`~plotnine.themes.themeable.legend_position_inside`).

        A tuple of values implies "inside" the panels at those exact values,
        which should be in the range `[0, 1]` within the panels area.

        A value of `"none"` turns off the legend.
    """