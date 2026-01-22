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
class legend_justification_right(themeable):
    """
    Justification of legends placed on the right

    Parameters
    ----------
    theme_element : Literal["bottom", "center", "top"] | float
        How to justify the entire group with 1 or more guides. i.e. How
        to slide the legend along the right column.
        If a float, it should be in the range `[0, 1]`, where
        `0` is `"bottom"` and `1` is `"top"`.
    """