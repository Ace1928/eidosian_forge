from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Callable, Sequence
from ..core.types import ID
from ..util.tornado import _CallbackGroup
def add_session_callbacks(self, callbacks: Sequence[SessionCallback]) -> None:
    """

        """
    for cb in callbacks:
        self.add_session_callback(cb)