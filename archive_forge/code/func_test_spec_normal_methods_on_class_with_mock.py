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
def test_spec_normal_methods_on_class_with_mock(self):
    mock = Mock(AsyncClass)
    self.assertIsInstance(mock.async_method, AsyncMock)
    self.assertIsInstance(mock.normal_method, Mock)