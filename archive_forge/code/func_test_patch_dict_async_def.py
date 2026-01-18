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
def test_patch_dict_async_def(self):
    foo = {'a': 'a'}

    @patch.dict(foo, {'a': 'b'})
    async def test_async():
        self.assertEqual(foo['a'], 'b')
    self.assertTrue(iscoroutinefunction(test_async))
    run(test_async())