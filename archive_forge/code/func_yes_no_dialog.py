from __future__ import annotations
import functools
from asyncio import get_running_loop
from typing import Any, Callable, Sequence, TypeVar
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.eventloop import run_in_executor_with_context
from prompt_toolkit.filters import FilterOrBool
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.key_binding.key_bindings import KeyBindings, merge_key_bindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import AnyContainer, HSplit
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.styles import BaseStyle
from prompt_toolkit.validation import Validator
from prompt_toolkit.widgets import (
def yes_no_dialog(title: AnyFormattedText='', text: AnyFormattedText='', yes_text: str='Yes', no_text: str='No', style: BaseStyle | None=None) -> Application[bool]:
    """
    Display a Yes/No dialog.
    Return a boolean.
    """

    def yes_handler() -> None:
        get_app().exit(result=True)

    def no_handler() -> None:
        get_app().exit(result=False)
    dialog = Dialog(title=title, body=Label(text=text, dont_extend_height=True), buttons=[Button(text=yes_text, handler=yes_handler), Button(text=no_text, handler=no_handler)], with_background=True)
    return _create_app(dialog, style)