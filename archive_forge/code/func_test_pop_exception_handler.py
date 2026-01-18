import unittest
from unittest import mock
from traits.observation.exception_handling import (
def test_pop_exception_handler(self):
    stack = ObserverExceptionHandlerStack()
    stack.push_exception_handler(reraise_exceptions=True)
    stack.pop_exception_handler()
    with mock.patch('sys.stderr'):
        try:
            raise ZeroDivisionError()
        except Exception:
            stack.handle_exception('Event')