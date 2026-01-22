from __future__ import annotations
import typing
from copy import copy, deepcopy
from functools import cached_property
from typing import overload
from ..exceptions import PlotnineError
from ..options import get_option, set_option
from .targets import ThemeTargets
from .themeable import Themeables, themeable
def _add_default_themeable_properties(self):
    """
        Add default themeable properties that depend depend on the plot

        Some properties may be left unset (None) and their final values are
        best worked out dynamically after the plot has been built, but
        before the themeables are applied.

        This is where the theme is modified to add those values.
        """
    self._smart_title_and_subtitle_ha()