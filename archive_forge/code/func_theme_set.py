from __future__ import annotations
import typing
from copy import copy, deepcopy
from functools import cached_property
from typing import overload
from ..exceptions import PlotnineError
from ..options import get_option, set_option
from .targets import ThemeTargets
from .themeable import Themeables, themeable
def theme_set(new: theme | Type[theme]) -> theme:
    """
    Change the current(default) theme

    Parameters
    ----------
    new : theme
        New default theme

    Returns
    -------
    out : theme
        Previous theme
    """
    if not isinstance(new, theme) and (not issubclass(new, theme)):
        raise PlotnineError('Expecting object to be a theme')
    out: theme = get_option('current_theme')
    set_option('current_theme', new)
    return out