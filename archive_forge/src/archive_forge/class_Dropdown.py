from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Callable
from ...core.enums import ButtonType
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from ...events import ButtonClick, MenuItemClick
from ..callbacks import Callback
from ..dom import DOMNode
from ..ui.icons import BuiltinIcon, Icon
from ..ui.tooltips import Tooltip
from .widget import Widget
class Dropdown(AbstractButton):
    """ A dropdown button.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    label = Override(default='Dropdown')
    split = Bool(default=False, help='\n    ')
    menu = List(Nullable(Either(String, Tuple(String, Either(String, Instance(Callback))))), help="\n    Button's dropdown menu consisting of entries containing item's text and\n    value name. Use ``None`` as a menu separator.\n    ")

    def on_click(self, handler: EventCallback) -> None:
        """ Set up a handler for button or menu item clicks.

        Args:
            handler (func) : handler function to call when button is activated.

        Returns:
            None

        """
        self.on_event(ButtonClick, handler)
        self.on_event(MenuItemClick, handler)

    def js_on_click(self, handler: Callback) -> None:
        """ Set up a JavaScript handler for button or menu item clicks. """
        self.js_on_event(ButtonClick, handler)
        self.js_on_event(MenuItemClick, handler)