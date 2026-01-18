from __future__ import annotations
from typing import TYPE_CHECKING, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import memoized
from prompt_toolkit.enums import EditingMode
from .base import Condition
@Condition
def vi_replace_single_mode() -> bool:
    from prompt_toolkit.key_binding.vi_state import InputMode
    app = get_app()
    if app.editing_mode != EditingMode.VI or app.vi_state.operator_func or app.vi_state.waiting_for_digraph or app.current_buffer.selection_state or app.vi_state.temporary_navigation_mode or app.current_buffer.read_only():
        return False
    return app.vi_state.input_mode == InputMode.REPLACE_SINGLE