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
def test_spec_mock_type_positional(self):

    def inner_test(mock_type):
        async_mock = mock_type(async_func)
        self.assertIsInstance(async_mock, mock_type)
        with assertNeverAwaited(self):
            self.assertTrue(inspect.isawaitable(async_mock()))
        sync_mock = mock_type(normal_func)
        self.assertIsInstance(sync_mock, mock_type)
    for mock_type in [AsyncMock, MagicMock, Mock]:
        with self.subTest(f'test spec positional with {mock_type}'):
            inner_test(mock_type)