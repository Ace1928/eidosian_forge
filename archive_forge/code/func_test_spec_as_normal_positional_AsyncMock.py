import asyncio
import gc
import inspect
import re
import unittest
from contextlib import contextmanager
from test import support
from asyncio import run, iscoroutinefunction
from unittest import IsolatedAsyncioTestCase
from unittest.mock import (ANY, call, AsyncMock, patch, MagicMock, Mock,
def test_spec_as_normal_positional_AsyncMock(self):
    mock = AsyncMock(normal_func)
    self.assertIsInstance(mock, AsyncMock)
    m = mock()
    self.assertTrue(inspect.isawaitable(m))
    run(m)