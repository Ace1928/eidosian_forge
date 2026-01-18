import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_spec_function_assert_has_calls(self):

    def f(a):
        pass
    mock = create_autospec(f)
    mock(1)
    mock.assert_has_calls([call(1)])
    with self.assertRaises(AssertionError):
        mock.assert_has_calls([call(2)])