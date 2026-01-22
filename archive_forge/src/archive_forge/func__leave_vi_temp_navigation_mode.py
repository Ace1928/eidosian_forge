from __future__ import annotations
import weakref
from asyncio import Task, sleep
from collections import deque
from typing import TYPE_CHECKING, Any, Generator
from prompt_toolkit.application.current import get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters.app import vi_navigation_mode
from prompt_toolkit.keys import Keys
from prompt_toolkit.utils import Event
from .key_bindings import Binding, KeyBindingsBase
def _leave_vi_temp_navigation_mode(self, event: KeyPressEvent) -> None:
    """
        If we're in Vi temporary navigation (normal) mode, return to
        insert/replace mode after executing one action.
        """
    app = event.app
    if app.editing_mode == EditingMode.VI:
        if app.vi_state.operator_func is None and self.arg is None:
            app.vi_state.temporary_navigation_mode = False