import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_spec_inspect_signature(self):

    def myfunc(x, y):
        pass
    mock = create_autospec(myfunc)
    mock(1, 2)
    mock(x=1, y=2)
    self.assertEqual(inspect.signature(mock), inspect.signature(myfunc))
    self.assertEqual(mock.mock_calls, [call(1, 2), call(x=1, y=2)])
    self.assertRaises(TypeError, mock, 1)