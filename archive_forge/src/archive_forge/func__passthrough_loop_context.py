import asyncio
import contextlib
import warnings
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional, Type, Union
import pytest
from aiohttp.helpers import isasyncgenfunction
from aiohttp.web import Application
from .test_utils import (
@contextlib.contextmanager
def _passthrough_loop_context(loop, fast=False):
    """Passthrough loop context.

    Sets up and tears down a loop unless one is passed in via the loop
    argument when it's passed straight through.
    """
    if loop:
        yield loop
    else:
        loop = setup_test_loop()
        yield loop
        teardown_test_loop(loop, fast=fast)