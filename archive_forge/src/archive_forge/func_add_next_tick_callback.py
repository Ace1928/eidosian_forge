from __future__ import annotations
import logging # isort:skip
import sys
import threading
from collections import defaultdict
from traceback import format_exception
from typing import (
import tornado
from tornado import gen
from ..core.types import ID
def add_next_tick_callback(self, callback: Callback, callback_id: ID) -> ID:
    """ Adds a callback to be run on the nex

        The passed-in ID can be used with remove_next_tick_callback.

        """

    def wrapper() -> None | Awaitable[None]:
        if wrapper.removed:
            return None
        self.remove_next_tick_callback(callback_id)
        return callback()
    wrapper.removed = False

    def remover() -> None:
        wrapper.removed = True
    self._assign_remover(callback, callback_id, self._next_tick_callback_removers, remover)
    self._loop.add_callback(wrapper)
    return callback_id