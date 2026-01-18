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
def test_target_not_async_spec_is(self):

    @patch.object(NormalClass, 'a', spec=async_func)
    def test_attribute_not_async_spec_is(mock_async_func):
        self.assertIsInstance(mock_async_func, AsyncMock)
    test_attribute_not_async_spec_is()