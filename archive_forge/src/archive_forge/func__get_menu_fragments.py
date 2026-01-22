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
def _get_menu_fragments(self) -> StyleAndTextTuples:
    focused = get_app().layout.has_focus(self.window)
    if not focused:
        self.selected_menu = [0]

    def one_item(i: int, item: MenuItem) -> Iterable[OneStyleAndTextTuple]:

        def mouse_handler(mouse_event: MouseEvent) -> None:
            hover = mouse_event.event_type == MouseEventType.MOUSE_MOVE
            if mouse_event.event_type == MouseEventType.MOUSE_DOWN or (hover and focused):
                app = get_app()
                if not hover:
                    if app.layout.has_focus(self.window):
                        if self.selected_menu == [i]:
                            app.layout.focus_last()
                    else:
                        app.layout.focus(self.window)
                self.selected_menu = [i]
        yield ('class:menu-bar', ' ', mouse_handler)
        if i == self.selected_menu[0] and focused:
            yield ('[SetMenuPosition]', '', mouse_handler)
            style = 'class:menu-bar.selected-item'
        else:
            style = 'class:menu-bar'
        yield (style, item.text, mouse_handler)
    result: StyleAndTextTuples = []
    for i, item in enumerate(self.menu_items):
        result.extend(one_item(i, item))
    return result