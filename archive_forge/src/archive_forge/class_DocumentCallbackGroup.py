from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Callable, Sequence
from ..core.types import ID
from ..util.tornado import _CallbackGroup
class DocumentCallbackGroup:
    """

    """

    def __init__(self, io_loop: IOLoop) -> None:
        """

        """
        self._group = _CallbackGroup(io_loop)

    def remove_all_callbacks(self) -> None:
        """

        """
        self._group.remove_all_callbacks()

    def add_session_callbacks(self, callbacks: Sequence[SessionCallback]) -> None:
        """

        """
        for cb in callbacks:
            self.add_session_callback(cb)

    def add_session_callback(self, callback_obj: SessionCallback) -> None:
        """

        """
        if isinstance(callback_obj, PeriodicCallback):
            self._group.add_periodic_callback(callback_obj.callback, callback_obj.period, callback_obj.id)
        elif isinstance(callback_obj, TimeoutCallback):
            self._group.add_timeout_callback(callback_obj.callback, callback_obj.timeout, callback_obj.id)
        elif isinstance(callback_obj, NextTickCallback):
            self._group.add_next_tick_callback(callback_obj.callback, callback_obj.id)
        else:
            raise ValueError(f'Expected callback of type PeriodicCallback, TimeoutCallback, NextTickCallback, got: {callback_obj.callback}')

    def remove_session_callback(self, callback_obj: SessionCallback) -> None:
        """

        """
        try:
            if isinstance(callback_obj, PeriodicCallback):
                self._group.remove_periodic_callback(callback_obj.id)
            elif isinstance(callback_obj, TimeoutCallback):
                self._group.remove_timeout_callback(callback_obj.id)
            elif isinstance(callback_obj, NextTickCallback):
                self._group.remove_next_tick_callback(callback_obj.id)
        except ValueError:
            pass