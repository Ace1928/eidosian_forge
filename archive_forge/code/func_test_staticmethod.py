import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_staticmethod(self):

    class WithStaticMethod:

        @staticmethod
        def staticfunc():
            pass
    self.assertTrue(_callable(WithStaticMethod.staticfunc))