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
def test_set_return_value(mock_type):
    mock_instance = mock_type(self.WithAsyncIterator())
    mock_instance.__aiter__.return_value = expected[:]
    self.assertEqual(run(iterate(mock_instance)), expected)