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
class legend_title_position(themeable):
    """
    Position of legend title

    Parameters
    ----------
    theme_element : Literal["top", "bottom", "left", "right"] | None
        Position of the legend title. The default depends on the position
        of the legend.
    """