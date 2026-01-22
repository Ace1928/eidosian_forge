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
class AsyncArguments(IsolatedAsyncioTestCase):

    async def test_add_return_value(self):

        async def addition(self, var):
            pass
        mock = AsyncMock(addition, return_value=10)
        output = await mock(5)
        self.assertEqual(output, 10)

    async def test_add_side_effect_exception(self):

        class CustomError(Exception):
            pass

        async def addition(var):
            pass
        mock = AsyncMock(addition, side_effect=CustomError('side-effect'))
        with self.assertRaisesRegex(CustomError, 'side-effect'):
            await mock(5)

    async def test_add_side_effect_coroutine(self):

        async def addition(var):
            return var + 1
        mock = AsyncMock(side_effect=addition)
        result = await mock(5)
        self.assertEqual(result, 6)

    async def test_add_side_effect_normal_function(self):

        def addition(var):
            return var + 1
        mock = AsyncMock(side_effect=addition)
        result = await mock(5)
        self.assertEqual(result, 6)

    async def test_add_side_effect_iterable(self):
        vals = [1, 2, 3]
        mock = AsyncMock(side_effect=vals)
        for item in vals:
            self.assertEqual(await mock(), item)
        with self.assertRaises(StopAsyncIteration) as e:
            await mock()

    async def test_add_side_effect_exception_iterable(self):

        class SampleException(Exception):
            pass
        vals = [1, SampleException('foo')]
        mock = AsyncMock(side_effect=vals)
        self.assertEqual(await mock(), 1)
        with self.assertRaises(SampleException) as e:
            await mock()

    async def test_return_value_AsyncMock(self):
        value = AsyncMock(return_value=10)
        mock = AsyncMock(return_value=value)
        result = await mock()
        self.assertIs(result, value)

    async def test_return_value_awaitable(self):
        fut = asyncio.Future()
        fut.set_result(None)
        mock = AsyncMock(return_value=fut)
        result = await mock()
        self.assertIsInstance(result, asyncio.Future)

    async def test_side_effect_awaitable_values(self):
        fut = asyncio.Future()
        fut.set_result(None)
        mock = AsyncMock(side_effect=[fut])
        result = await mock()
        self.assertIsInstance(result, asyncio.Future)
        with self.assertRaises(StopAsyncIteration):
            await mock()

    async def test_side_effect_is_AsyncMock(self):
        effect = AsyncMock(return_value=10)
        mock = AsyncMock(side_effect=effect)
        result = await mock()
        self.assertEqual(result, 10)

    async def test_wraps_coroutine(self):
        value = asyncio.Future()
        ran = False

        async def inner():
            nonlocal ran
            ran = True
            return value
        mock = AsyncMock(wraps=inner)
        result = await mock()
        self.assertEqual(result, value)
        mock.assert_awaited()
        self.assertTrue(ran)

    async def test_wraps_normal_function(self):
        value = 1
        ran = False

        def inner():
            nonlocal ran
            ran = True
            return value
        mock = AsyncMock(wraps=inner)
        result = await mock()
        self.assertEqual(result, value)
        mock.assert_awaited()
        self.assertTrue(ran)

    async def test_await_args_list_order(self):
        async_mock = AsyncMock()
        mock2 = async_mock(2)
        mock1 = async_mock(1)
        await mock1
        await mock2
        async_mock.assert_has_awaits([call(1), call(2)])
        self.assertEqual(async_mock.await_args_list, [call(1), call(2)])
        self.assertEqual(async_mock.call_args_list, [call(2), call(1)])