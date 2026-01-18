from __future__ import annotations
import typing
from copy import copy, deepcopy
from functools import cached_property
from typing import overload
from ..exceptions import PlotnineError
from ..options import get_option, set_option
from .targets import ThemeTargets
from .themeable import Themeables, themeable
def theme_get() -> theme:
    """
    Return the default theme

    The default theme is the one set (using [](`~plotnine.themes.theme_set`))
    by the user. If none has been set, then [](`~plotnine.themes.theme_gray`)
    is the default.
    """
    from .theme_gray import theme_gray
    _theme = get_option('current_theme')
    if isinstance(_theme, type):
        _theme = _theme()
    return _theme or theme_gray()