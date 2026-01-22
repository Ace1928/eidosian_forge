import asyncio
from collections.abc import Generator
import functools
import inspect
import logging
import os
import re
import signal
import socket
import sys
import unittest
import warnings
from tornado import gen
from tornado.httpclient import AsyncHTTPClient, HTTPResponse
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop, TimeoutError
from tornado import netutil
from tornado.platform.asyncio import AsyncIOMainLoop
from tornado.process import Subprocess
from tornado.log import app_log
from tornado.util import raise_exc_info, basestring_type
from tornado.web import Application
import typing
from typing import Tuple, Any, Callable, Type, Dict, Union, Optional, Coroutine
from types import TracebackType
class _TestMethodWrapper(object):
    """Wraps a test method to raise an error if it returns a value.

    This is mainly used to detect undecorated generators (if a test
    method yields it must use a decorator to consume the generator),
    but will also detect other kinds of return values (these are not
    necessarily errors, but we alert anyway since there is no good
    reason to return a value from a test).
    """

    def __init__(self, orig_method: Callable) -> None:
        self.orig_method = orig_method
        self.__wrapped__ = orig_method

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        result = self.orig_method(*args, **kwargs)
        if isinstance(result, Generator) or inspect.iscoroutine(result):
            raise TypeError('Generator and coroutine test methods should be decorated with tornado.testing.gen_test')
        elif result is not None:
            raise ValueError('Return value from test method ignored: %r' % result)

    def __getattr__(self, name: str) -> Any:
        """Proxy all unknown attributes to the original method.

        This is important for some of the decorators in the `unittest`
        module, such as `unittest.skipIf`.
        """
        return getattr(self.orig_method, name)