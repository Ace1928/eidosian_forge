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
class rect(legend_key, legend_frame, legend_background, panel_background, panel_border, plot_background, strip_background):
    """
    All rectangle elements

    Parameters
    ----------
    theme_element : element_rect
    """