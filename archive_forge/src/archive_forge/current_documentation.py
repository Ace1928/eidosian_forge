from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Generator

    Create `AppSession` that always prefers the TTY input/output.

    Even if `sys.stdin` and `sys.stdout` are connected to input/output pipes,
    this will still use the terminal for interaction (because `sys.stderr` is
    still connected to the terminal).

    Usage::

        from prompt_toolkit.shortcuts import prompt

        with create_app_session_from_tty():
            prompt('>')
    