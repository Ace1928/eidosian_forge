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
class AsyncMockAssert(unittest.TestCase):

    def setUp(self):
        self.mock = AsyncMock()

    async def _runnable_test(self, *args, **kwargs):
        await self.mock(*args, **kwargs)

    async def _await_coroutine(self, coroutine):
        return await coroutine

    def test_assert_called_but_not_awaited(self):
        mock = AsyncMock(AsyncClass)
        with assertNeverAwaited(self):
            mock.async_method()
        self.assertTrue(iscoroutinefunction(mock.async_method))
        mock.async_method.assert_called()
        mock.async_method.assert_called_once()
        mock.async_method.assert_called_once_with()
        with self.assertRaises(AssertionError):
            mock.assert_awaited()
        with self.assertRaises(AssertionError):
            mock.async_method.assert_awaited()

    def test_assert_called_then_awaited(self):
        mock = AsyncMock(AsyncClass)
        mock_coroutine = mock.async_method()
        mock.async_method.assert_called()
        mock.async_method.assert_called_once()
        mock.async_method.assert_called_once_with()
        with self.assertRaises(AssertionError):
            mock.async_method.assert_awaited()
        run(self._await_coroutine(mock_coroutine))
        mock.async_method.assert_called_once()
        mock.async_method.assert_awaited()
        mock.async_method.assert_awaited_once()
        mock.async_method.assert_awaited_once_with()

    def test_assert_called_and_awaited_at_same_time(self):
        with self.assertRaises(AssertionError):
            self.mock.assert_awaited()
        with self.assertRaises(AssertionError):
            self.mock.assert_called()
        run(self._runnable_test())
        self.mock.assert_called_once()
        self.mock.assert_awaited_once()

    def test_assert_called_twice_and_awaited_once(self):
        mock = AsyncMock(AsyncClass)
        coroutine = mock.async_method()
        with assertNeverAwaited(self):
            mock.async_method()
        with self.assertRaises(AssertionError):
            mock.async_method.assert_awaited()
        mock.async_method.assert_called()
        run(self._await_coroutine(coroutine))
        mock.async_method.assert_awaited()
        mock.async_method.assert_awaited_once()

    def test_assert_called_once_and_awaited_twice(self):
        mock = AsyncMock(AsyncClass)
        coroutine = mock.async_method()
        mock.async_method.assert_called_once()
        run(self._await_coroutine(coroutine))
        with self.assertRaises(RuntimeError):
            run(self._await_coroutine(coroutine))
        mock.async_method.assert_awaited()

    def test_assert_awaited_but_not_called(self):
        with self.assertRaises(AssertionError):
            self.mock.assert_awaited()
        with self.assertRaises(AssertionError):
            self.mock.assert_called()
        with self.assertRaises(TypeError):
            run(self._await_coroutine(self.mock))
        with self.assertRaises(AssertionError):
            self.mock.assert_awaited()
        with self.assertRaises(AssertionError):
            self.mock.assert_called()

    def test_assert_has_calls_not_awaits(self):
        kalls = [call('foo')]
        with assertNeverAwaited(self):
            self.mock('foo')
        self.mock.assert_has_calls(kalls)
        with self.assertRaises(AssertionError):
            self.mock.assert_has_awaits(kalls)

    def test_assert_has_mock_calls_on_async_mock_no_spec(self):
        with assertNeverAwaited(self):
            self.mock()
        kalls_empty = [('', (), {})]
        self.assertEqual(self.mock.mock_calls, kalls_empty)
        with assertNeverAwaited(self):
            self.mock('foo')
        with assertNeverAwaited(self):
            self.mock('baz')
        mock_kalls = [call(), call('foo'), call('baz')]
        self.assertEqual(self.mock.mock_calls, mock_kalls)

    def test_assert_has_mock_calls_on_async_mock_with_spec(self):
        a_class_mock = AsyncMock(AsyncClass)
        with assertNeverAwaited(self):
            a_class_mock.async_method()
        kalls_empty = [('', (), {})]
        self.assertEqual(a_class_mock.async_method.mock_calls, kalls_empty)
        self.assertEqual(a_class_mock.mock_calls, [call.async_method()])
        with assertNeverAwaited(self):
            a_class_mock.async_method(1, 2, 3, a=4, b=5)
        method_kalls = [call(), call(1, 2, 3, a=4, b=5)]
        mock_kalls = [call.async_method(), call.async_method(1, 2, 3, a=4, b=5)]
        self.assertEqual(a_class_mock.async_method.mock_calls, method_kalls)
        self.assertEqual(a_class_mock.mock_calls, mock_kalls)

    def test_async_method_calls_recorded(self):
        with assertNeverAwaited(self):
            self.mock.something(3, fish=None)
        with assertNeverAwaited(self):
            self.mock.something_else.something(6, cake=sentinel.Cake)
        self.assertEqual(self.mock.method_calls, [('something', (3,), {'fish': None}), ('something_else.something', (6,), {'cake': sentinel.Cake})], 'method calls not recorded correctly')
        self.assertEqual(self.mock.something_else.method_calls, [('something', (6,), {'cake': sentinel.Cake})], 'method calls not recorded correctly')

    def test_async_arg_lists(self):

        def assert_attrs(mock):
            names = ('call_args_list', 'method_calls', 'mock_calls')
            for name in names:
                attr = getattr(mock, name)
                self.assertIsInstance(attr, _CallList)
                self.assertIsInstance(attr, list)
                self.assertEqual(attr, [])
        assert_attrs(self.mock)
        with assertNeverAwaited(self):
            self.mock()
        with assertNeverAwaited(self):
            self.mock(1, 2)
        with assertNeverAwaited(self):
            self.mock(a=3)
        self.mock.reset_mock()
        assert_attrs(self.mock)
        a_mock = AsyncMock(AsyncClass)
        with assertNeverAwaited(self):
            a_mock.async_method()
        with assertNeverAwaited(self):
            a_mock.async_method(1, a=3)
        a_mock.reset_mock()
        assert_attrs(a_mock)

    def test_assert_awaited(self):
        with self.assertRaises(AssertionError):
            self.mock.assert_awaited()
        run(self._runnable_test())
        self.mock.assert_awaited()

    def test_assert_awaited_once(self):
        with self.assertRaises(AssertionError):
            self.mock.assert_awaited_once()
        run(self._runnable_test())
        self.mock.assert_awaited_once()
        run(self._runnable_test())
        with self.assertRaises(AssertionError):
            self.mock.assert_awaited_once()

    def test_assert_awaited_with(self):
        msg = 'Not awaited'
        with self.assertRaisesRegex(AssertionError, msg):
            self.mock.assert_awaited_with('foo')
        run(self._runnable_test())
        msg = 'expected await not found'
        with self.assertRaisesRegex(AssertionError, msg):
            self.mock.assert_awaited_with('foo')
        run(self._runnable_test('foo'))
        self.mock.assert_awaited_with('foo')
        run(self._runnable_test('SomethingElse'))
        with self.assertRaises(AssertionError):
            self.mock.assert_awaited_with('foo')

    def test_assert_awaited_once_with(self):
        with self.assertRaises(AssertionError):
            self.mock.assert_awaited_once_with('foo')
        run(self._runnable_test('foo'))
        self.mock.assert_awaited_once_with('foo')
        run(self._runnable_test('foo'))
        with self.assertRaises(AssertionError):
            self.mock.assert_awaited_once_with('foo')

    def test_assert_any_wait(self):
        with self.assertRaises(AssertionError):
            self.mock.assert_any_await('foo')
        run(self._runnable_test('baz'))
        with self.assertRaises(AssertionError):
            self.mock.assert_any_await('foo')
        run(self._runnable_test('foo'))
        self.mock.assert_any_await('foo')
        run(self._runnable_test('SomethingElse'))
        self.mock.assert_any_await('foo')

    def test_assert_has_awaits_no_order(self):
        calls = [call('foo'), call('baz')]
        with self.assertRaises(AssertionError) as cm:
            self.mock.assert_has_awaits(calls)
        self.assertEqual(len(cm.exception.args), 1)
        run(self._runnable_test('foo'))
        with self.assertRaises(AssertionError):
            self.mock.assert_has_awaits(calls)
        run(self._runnable_test('foo'))
        with self.assertRaises(AssertionError):
            self.mock.assert_has_awaits(calls)
        run(self._runnable_test('baz'))
        self.mock.assert_has_awaits(calls)
        run(self._runnable_test('SomethingElse'))
        self.mock.assert_has_awaits(calls)

    def test_awaits_asserts_with_any(self):

        class Foo:

            def __eq__(self, other):
                pass
        run(self._runnable_test(Foo(), 1))
        self.mock.assert_has_awaits([call(ANY, 1)])
        self.mock.assert_awaited_with(ANY, 1)
        self.mock.assert_any_await(ANY, 1)

    def test_awaits_asserts_with_spec_and_any(self):

        class Foo:

            def __eq__(self, other):
                pass
        mock_with_spec = AsyncMock(spec=Foo)

        async def _custom_mock_runnable_test(*args):
            await mock_with_spec(*args)
        run(_custom_mock_runnable_test(Foo(), 1))
        mock_with_spec.assert_has_awaits([call(ANY, 1)])
        mock_with_spec.assert_awaited_with(ANY, 1)
        mock_with_spec.assert_any_await(ANY, 1)

    def test_assert_has_awaits_ordered(self):
        calls = [call('foo'), call('baz')]
        with self.assertRaises(AssertionError):
            self.mock.assert_has_awaits(calls, any_order=True)
        run(self._runnable_test('baz'))
        with self.assertRaises(AssertionError):
            self.mock.assert_has_awaits(calls, any_order=True)
        run(self._runnable_test('bamf'))
        with self.assertRaises(AssertionError):
            self.mock.assert_has_awaits(calls, any_order=True)
        run(self._runnable_test('foo'))
        self.mock.assert_has_awaits(calls, any_order=True)
        run(self._runnable_test('qux'))
        self.mock.assert_has_awaits(calls, any_order=True)

    def test_assert_not_awaited(self):
        self.mock.assert_not_awaited()
        run(self._runnable_test())
        with self.assertRaises(AssertionError):
            self.mock.assert_not_awaited()

    def test_assert_has_awaits_not_matching_spec_error(self):

        async def f(x=None):
            pass
        self.mock = AsyncMock(spec=f)
        run(self._runnable_test(1))
        with self.assertRaisesRegex(AssertionError, '^{}$'.format(re.escape('Awaits not found.\nExpected: [call()]\nActual: [call(1)]'))) as cm:
            self.mock.assert_has_awaits([call()])
        self.assertIsNone(cm.exception.__cause__)
        with self.assertRaisesRegex(AssertionError, '^{}$'.format(re.escape("Error processing expected awaits.\nErrors: [None, TypeError('too many positional arguments')]\nExpected: [call(), call(1, 2)]\nActual: [call(1)]"))) as cm:
            self.mock.assert_has_awaits([call(), call(1, 2)])
        self.assertIsInstance(cm.exception.__cause__, TypeError)