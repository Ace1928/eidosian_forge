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
def test_spec_normal_methods_on_class_with_mock_seal(self):
    mock = Mock(AsyncClass)
    seal(mock)
    with self.assertRaises(AttributeError):
        mock.normal_method
    with self.assertRaises(AttributeError):
        mock.async_method