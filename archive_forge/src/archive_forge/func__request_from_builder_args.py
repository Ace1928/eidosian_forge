from __future__ import annotations
import importlib.metadata
import typing as t
from contextlib import contextmanager
from contextlib import ExitStack
from copy import copy
from types import TracebackType
from urllib.parse import urlsplit
import werkzeug.test
from click.testing import CliRunner
from werkzeug.test import Client
from werkzeug.wrappers import Request as BaseRequest
from .cli import ScriptInfo
from .sessions import SessionMixin
def _request_from_builder_args(self, args: tuple[t.Any, ...], kwargs: dict[str, t.Any]) -> BaseRequest:
    kwargs['environ_base'] = self._copy_environ(kwargs.get('environ_base', {}))
    builder = EnvironBuilder(self.application, *args, **kwargs)
    try:
        return builder.get_request()
    finally:
        builder.close()