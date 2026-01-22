from __future__ import annotations
import logging # isort:skip
from ...core.has_props import abstract
from ...core.properties import (
from .buttons import ButtonLike
from .widget import Widget
class CheckboxButtonGroup(ToggleButtonGroup):
    """ A group of check boxes rendered as toggle buttons.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    active = List(Int, help='\n    The list of indices of selected check boxes.\n    ')