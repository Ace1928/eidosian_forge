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
def _start_timeout(self) -> None:
    """
        Start auto flush timeout. Similar to Vim's `timeoutlen` option.

        Start a background coroutine with a timer. When this timeout expires
        and no key was pressed in the meantime, we flush all data in the queue
        and call the appropriate key binding handlers.
        """
    app = get_app()
    timeout = app.timeoutlen
    if timeout is None:
        return

    async def wait() -> None:
        """Wait for timeout."""
        await sleep(timeout)
        if len(self.key_buffer) > 0:
            flush_keys()

    def flush_keys() -> None:
        """Flush keys."""
        self.feed(_Flush)
        self.process_keys()
    if self._flush_wait_task:
        self._flush_wait_task.cancel()
    self._flush_wait_task = app.create_background_task(wait())