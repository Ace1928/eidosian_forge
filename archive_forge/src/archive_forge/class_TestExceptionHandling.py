import unittest
from unittest import mock
from traits.observation.exception_handling import (
class TestExceptionHandling(unittest.TestCase):

    def test_default_logging(self):
        stack = ObserverExceptionHandlerStack()
        with self.assertLogs('traits', level='ERROR') as log_context:
            try:
                raise ZeroDivisionError()
            except Exception:
                stack.handle_exception('Event')
        content, = log_context.output
        self.assertIn('Exception occurred in traits notification handler for event object: {!r}'.format('Event'), content)

    def test_push_exception_handler(self):
        stack = ObserverExceptionHandlerStack()
        stack.push_exception_handler(reraise_exceptions=True)
        with self.assertLogs('traits', level='ERROR') as log_context, self.assertRaises(ZeroDivisionError):
            try:
                raise ZeroDivisionError()
            except Exception:
                stack.handle_exception('Event')
        content, = log_context.output
        self.assertIn('ZeroDivisionError', content)

    def test_push_exception_handler_collect_events(self):
        events = []

        def handler(event):
            events.append(event)
        stack = ObserverExceptionHandlerStack()
        stack.push_exception_handler(handler=handler)
        try:
            raise ZeroDivisionError()
        except Exception:
            stack.handle_exception('Event')
        self.assertEqual(events, ['Event'])

    def test_pop_exception_handler(self):
        stack = ObserverExceptionHandlerStack()
        stack.push_exception_handler(reraise_exceptions=True)
        stack.pop_exception_handler()
        with mock.patch('sys.stderr'):
            try:
                raise ZeroDivisionError()
            except Exception:
                stack.handle_exception('Event')