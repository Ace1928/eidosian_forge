from __future__ import annotations
from typing import Any
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.enums import SYSTEM_BUFFER
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.key_binding.key_bindings import (
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import ConditionalContainer, Container, Window
from prompt_toolkit.layout.controls import (
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.lexers import SimpleLexer
from prompt_toolkit.search import SearchDirection
def _build_key_bindings(self) -> KeyBindingsBase:
    focused = has_focus(self.system_buffer)
    emacs_bindings = KeyBindings()
    handle = emacs_bindings.add

    @handle('escape', filter=focused)
    @handle('c-g', filter=focused)
    @handle('c-c', filter=focused)
    def _cancel(event: E) -> None:
        """Hide system prompt."""
        self.system_buffer.reset()
        event.app.layout.focus_last()

    @handle('enter', filter=focused)
    async def _accept(event: E) -> None:
        """Run system command."""
        await event.app.run_system_command(self.system_buffer.text, display_before_text=self._get_display_before_text())
        self.system_buffer.reset(append_to_history=True)
        event.app.layout.focus_last()
    vi_bindings = KeyBindings()
    handle = vi_bindings.add

    @handle('escape', filter=focused)
    @handle('c-c', filter=focused)
    def _cancel_vi(event: E) -> None:
        """Hide system prompt."""
        event.app.vi_state.input_mode = InputMode.NAVIGATION
        self.system_buffer.reset()
        event.app.layout.focus_last()

    @handle('enter', filter=focused)
    async def _accept_vi(event: E) -> None:
        """Run system command."""
        event.app.vi_state.input_mode = InputMode.NAVIGATION
        await event.app.run_system_command(self.system_buffer.text, display_before_text=self._get_display_before_text())
        self.system_buffer.reset(append_to_history=True)
        event.app.layout.focus_last()
    global_bindings = KeyBindings()
    handle = global_bindings.add

    @handle(Keys.Escape, '!', filter=~focused & emacs_mode, is_global=True)
    def _focus_me(event: E) -> None:
        """M-'!' will focus this user control."""
        event.app.layout.focus(self.window)

    @handle('!', filter=~focused & vi_mode & vi_navigation_mode, is_global=True)
    def _focus_me_vi(event: E) -> None:
        """Focus."""
        event.app.vi_state.input_mode = InputMode.INSERT
        event.app.layout.focus(self.window)
    return merge_key_bindings([ConditionalKeyBindings(emacs_bindings, emacs_mode), ConditionalKeyBindings(vi_bindings, vi_mode), ConditionalKeyBindings(global_bindings, self.enable_global_bindings)])