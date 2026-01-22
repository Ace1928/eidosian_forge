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
class legend_box_just(themeable):
    """
    Justification of guide boxes

    Parameters
    ----------
    theme_element : Literal["left", "right", "center", "top", "bottom",                     "baseline"], default=None
        If `None`, the value that will apply depends on
        [](`~plotnine.theme.themeables.legend_box`).
    """