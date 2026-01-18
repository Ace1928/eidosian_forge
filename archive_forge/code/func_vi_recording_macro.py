from __future__ import annotations
from typing import TYPE_CHECKING, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import memoized
from prompt_toolkit.enums import EditingMode
from .base import Condition
@Condition
def vi_recording_macro() -> bool:
    """When recording a Vi macro."""
    app = get_app()
    if app.editing_mode != EditingMode.VI:
        return False
    return app.vi_state.recording_register is not None