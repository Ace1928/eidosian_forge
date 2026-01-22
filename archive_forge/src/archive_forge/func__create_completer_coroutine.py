from __future__ import annotations
import asyncio
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import deque
from enum import Enum
from functools import wraps
from typing import Any, Callable, Coroutine, Iterable, TypeVar, cast
from .application.current import get_app
from .application.run_in_terminal import run_in_terminal
from .auto_suggest import AutoSuggest, Suggestion
from .cache import FastDictCache
from .clipboard import ClipboardData
from .completion import (
from .document import Document
from .eventloop import aclosing
from .filters import FilterOrBool, to_filter
from .history import History, InMemoryHistory
from .search import SearchDirection, SearchState
from .selection import PasteMode, SelectionState, SelectionType
from .utils import Event, to_str
from .validation import ValidationError, Validator
def _create_completer_coroutine(self) -> Callable[..., Coroutine[Any, Any, None]]:
    """
        Create function for asynchronous autocompletion.

        (This consumes the asynchronous completer generator, which possibly
        runs the completion algorithm in another thread.)
        """

    def completion_does_nothing(document: Document, completion: Completion) -> bool:
        """
            Return `True` if applying this completion doesn't have any effect.
            (When it doesn't insert any new text.
            """
        text_before_cursor = document.text_before_cursor
        replaced_text = text_before_cursor[len(text_before_cursor) + completion.start_position:]
        return replaced_text == completion.text

    @_only_one_at_a_time
    async def async_completer(select_first: bool=False, select_last: bool=False, insert_common_part: bool=False, complete_event: CompleteEvent | None=None) -> None:
        document = self.document
        complete_event = complete_event or CompleteEvent(text_inserted=True)
        if self.complete_state or not self.completer:
            return
        complete_state = CompletionState(original_document=self.document)
        self.complete_state = complete_state

        def proceed() -> bool:
            """Keep retrieving completions. Input text has not yet changed
                while generating completions."""
            return self.complete_state == complete_state
        refresh_needed = asyncio.Event()

        async def refresh_while_loading() -> None:
            """Background loop to refresh the UI at most 3 times a second
                while the completion are loading. Calling
                `on_completions_changed.fire()` for every completion that we
                receive is too expensive when there are many completions. (We
                could tune `Application.max_render_postpone_time` and
                `Application.min_redraw_interval`, but having this here is a
                better approach.)
                """
            while True:
                self.on_completions_changed.fire()
                refresh_needed.clear()
                await asyncio.sleep(0.3)
                await refresh_needed.wait()
        refresh_task = asyncio.ensure_future(refresh_while_loading())
        try:
            async with aclosing(self.completer.get_completions_async(document, complete_event)) as async_generator:
                async for completion in async_generator:
                    complete_state.completions.append(completion)
                    refresh_needed.set()
                    if not proceed():
                        break
        finally:
            refresh_task.cancel()
            self.on_completions_changed.fire()
        completions = complete_state.completions
        if len(completions) == 1 and completion_does_nothing(document, completions[0]):
            del completions[:]
        if proceed():
            if not self.complete_state or self.complete_state.complete_index is not None:
                return
            if not completions:
                self.complete_state = None
                self.on_completions_changed.fire()
                return
            if select_first:
                self.go_to_completion(0)
            elif select_last:
                self.go_to_completion(len(completions) - 1)
            elif insert_common_part:
                common_part = get_common_complete_suffix(document, completions)
                if common_part:
                    self.insert_text(common_part)
                    if len(completions) > 1:
                        completions[:] = [c.new_completion_from_position(len(common_part)) for c in completions]
                        self._set_completions(completions=completions)
                    else:
                        self.complete_state = None
                elif len(completions) == 1:
                    self.go_to_completion(0)
        else:
            if self.document.text_before_cursor == document.text_before_cursor:
                return
            if self.document.text_before_cursor.startswith(document.text_before_cursor):
                raise _Retry
    return async_completer