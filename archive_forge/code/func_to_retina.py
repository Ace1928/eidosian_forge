from __future__ import annotations
import typing
from copy import copy, deepcopy
from functools import cached_property
from typing import overload
from ..exceptions import PlotnineError
from ..options import get_option, set_option
from .targets import ThemeTargets
from .themeable import Themeables, themeable
def to_retina(self) -> theme:
    """
        Return a retina-sized version of this theme

        The result is a theme that has double the dpi.
        """
    dpi = self.getp('dpi')
    return self + theme(dpi=dpi * 2)