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
class AsyncAutospecTest(unittest.TestCase):

    def test_is_AsyncMock_patch(self):

        @patch(async_foo_name, autospec=True)
        def test_async(mock_method):
            self.assertIsInstance(mock_method.async_method, AsyncMock)
            self.assertIsInstance(mock_method, MagicMock)

        @patch(async_foo_name, autospec=True)
        def test_normal_method(mock_method):
            self.assertIsInstance(mock_method.normal_method, MagicMock)
        test_async()
        test_normal_method()

    def test_create_autospec_instance(self):
        with self.assertRaises(RuntimeError):
            create_autospec(async_func, instance=True)

    @unittest.skip('Broken test from https://bugs.python.org/issue37251')
    def test_create_autospec_awaitable_class(self):
        self.assertIsInstance(create_autospec(AwaitableClass), AsyncMock)

    def test_create_autospec(self):
        spec = create_autospec(async_func_args)
        awaitable = spec(1, 2, c=3)

        async def main():
            await awaitable
        self.assertEqual(spec.await_count, 0)
        self.assertIsNone(spec.await_args)
        self.assertEqual(spec.await_args_list, [])
        spec.assert_not_awaited()
        run(main())
        self.assertTrue(iscoroutinefunction(spec))
        self.assertTrue(asyncio.iscoroutine(awaitable))
        self.assertEqual(spec.await_count, 1)
        self.assertEqual(spec.await_args, call(1, 2, c=3))
        self.assertEqual(spec.await_args_list, [call(1, 2, c=3)])
        spec.assert_awaited_once()
        spec.assert_awaited_once_with(1, 2, c=3)
        spec.assert_awaited_with(1, 2, c=3)
        spec.assert_awaited()
        with self.assertRaises(AssertionError):
            spec.assert_any_await(e=1)

    def test_patch_with_autospec(self):

        async def test_async():
            with patch(f'{__name__}.async_func_args', autospec=True) as mock_method:
                awaitable = mock_method(1, 2, c=3)
                self.assertIsInstance(mock_method.mock, AsyncMock)
                self.assertTrue(iscoroutinefunction(mock_method))
                self.assertTrue(asyncio.iscoroutine(awaitable))
                self.assertTrue(inspect.isawaitable(awaitable))
                self.assertEqual(mock_method.await_count, 0)
                self.assertEqual(mock_method.await_args_list, [])
                self.assertIsNone(mock_method.await_args)
                mock_method.assert_not_awaited()
                await awaitable
            self.assertEqual(mock_method.await_count, 1)
            self.assertEqual(mock_method.await_args, call(1, 2, c=3))
            self.assertEqual(mock_method.await_args_list, [call(1, 2, c=3)])
            mock_method.assert_awaited_once()
            mock_method.assert_awaited_once_with(1, 2, c=3)
            mock_method.assert_awaited_with(1, 2, c=3)
            mock_method.assert_awaited()
            mock_method.reset_mock()
            self.assertEqual(mock_method.await_count, 0)
            self.assertIsNone(mock_method.await_args)
            self.assertEqual(mock_method.await_args_list, [])
        run(test_async())