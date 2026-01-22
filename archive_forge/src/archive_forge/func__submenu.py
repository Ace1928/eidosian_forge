from __future__ import annotations
from typing import Callable, Iterable, Sequence
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text.base import OneStyleAndTextTuple, StyleAndTextTuples
from prompt_toolkit.key_binding.key_bindings import KeyBindings, KeyBindingsBase
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import (
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Shadow
from .base import Border
@kb.add('right', filter=in_sub_menu)
def _submenu(event: E) -> None:
    """go into sub menu."""
    if self._get_menu(len(self.selected_menu) - 1).children:
        self.selected_menu.append(0)
    elif len(self.selected_menu) == 2 and self.selected_menu[0] < len(self.menu_items) - 1:
        self.selected_menu = [min(len(self.menu_items) - 1, self.selected_menu[0] + 1)]
        if self.menu_items[self.selected_menu[0]].children:
            self.selected_menu.append(0)