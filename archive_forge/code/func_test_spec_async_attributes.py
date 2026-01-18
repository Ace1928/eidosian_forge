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
def test_spec_async_attributes(self):

    @patch(normal_foo_name, spec=AsyncClass)
    def test_async_attributes_coroutines(MockNormalClass):
        self.assertIsInstance(MockNormalClass.async_method, AsyncMock)
        self.assertIsInstance(MockNormalClass, MagicMock)
    test_async_attributes_coroutines()