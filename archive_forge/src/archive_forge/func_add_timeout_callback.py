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
def add_timeout_callback(self, callback: CallbackSync, timeout_milliseconds: int, callback_id: ID) -> ID:
    """ Adds a callback to be run once after timeout_milliseconds.

        The passed-in ID can be used with remove_timeout_callback.

        """

    def wrapper() -> None:
        self.remove_timeout_callback(callback_id)
        return callback()
    handle: object | None = None

    def remover() -> None:
        if handle is not None:
            self._loop.remove_timeout(handle)
    self._assign_remover(callback, callback_id, self._timeout_callback_removers, remover)
    handle = self._loop.call_later(timeout_milliseconds / 1000.0, wrapper)
    return callback_id