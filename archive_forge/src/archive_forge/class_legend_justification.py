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
class legend_justification(legend_justification_right, legend_justification_left, legend_justification_top, legend_justification_bottom, legend_justification_inside):
    """
    Justification of any legend

    Parameters
    ----------
    theme_element : Literal["left", "right", "center", "top", "bottom"] |                     float | tuple[float, float]
        How to justify the entire group with 1 or more guides.
    """