from __future__ import annotations
import asyncio
import contextvars
import socket
from asyncio import get_running_loop
from typing import Any, Callable, Coroutine, TextIO, cast
from prompt_toolkit.application.current import create_app_session, get_app
from prompt_toolkit.application.run_in_terminal import run_in_terminal
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.renderer import print_formatted_text as print_formatted_text
from prompt_toolkit.styles import BaseStyle, DummyStyle
from .log import logger
from .protocol import (
def ttype_received(ttype: str) -> None:
    """TelnetProtocolParser 'ttype_received' callback"""
    self.vt100_output = Vt100_Output(self.stdout, get_size, term=ttype, enable_cpr=enable_cpr)
    self._ready.set()