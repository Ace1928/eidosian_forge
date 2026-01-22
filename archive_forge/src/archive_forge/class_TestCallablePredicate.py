import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
class TestCallablePredicate(unittest.TestCase):

    def test_type(self):
        for obj in [str, bytes, int, list, tuple, SomeClass]:
            self.assertTrue(_callable(obj))

    def test_call_magic_method(self):

        class Callable:

            def __call__(self):
                pass
        instance = Callable()
        self.assertTrue(_callable(instance))

    def test_staticmethod(self):

        class WithStaticMethod:

            @staticmethod
            def staticfunc():
                pass
        self.assertTrue(_callable(WithStaticMethod.staticfunc))

    def test_non_callable_staticmethod(self):

        class BadStaticMethod:
            not_callable = staticmethod(None)
        self.assertFalse(_callable(BadStaticMethod.not_callable))

    def test_classmethod(self):

        class WithClassMethod:

            @classmethod
            def classfunc(cls):
                pass
        self.assertTrue(_callable(WithClassMethod.classfunc))

    def test_non_callable_classmethod(self):

        class BadClassMethod:
            not_callable = classmethod(None)
        self.assertFalse(_callable(BadClassMethod.not_callable))