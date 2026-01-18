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
def test_spec_normal_methods_on_class(self):

    def inner_test(mock_type):
        mock = mock_type(AsyncClass)
        self.assertIsInstance(mock.async_method, AsyncMock)
        self.assertIsInstance(mock.normal_method, MagicMock)
    for mock_type in [AsyncMock, MagicMock]:
        with self.subTest(f'test method types with {mock_type}'):
            inner_test(mock_type)