import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_spec_has_function_not_in_bases(self):

    class CrazyClass(object):

        def __dir__(self):
            return super(CrazyClass, self).__dir__() + ['crazy']

        def __getattr__(self, item):
            if item == 'crazy':
                return lambda x: x
            raise AttributeError(item)
    inst = CrazyClass()
    with self.assertRaises(AttributeError):
        inst.other
    self.assertEqual(inst.crazy(42), 42)
    mock = create_autospec(inst)
    mock.crazy(42)
    with self.assertRaises(TypeError):
        mock.crazy()
    with self.assertRaises(TypeError):
        mock.crazy(1, 2)