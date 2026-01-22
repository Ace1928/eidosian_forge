from __future__ import annotations
import logging # isort:skip
from ...core.enums import ToolIcon
from ...core.has_props import abstract
from ...core.properties import (
from ...model import Model
from ..callbacks import Callback
from .ui_element import UIElement
class CheckableItem(ActionItem):
    """ A two state checkable menu item. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    checked = Bool(default=False, help='\n    The state of the checkable item.\n\n    Checked item is represented with a tick mark on the left hand side\n    of an item. Unchecked item is represented with an empty space.\n    ')