import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class GetClassTests(TestCase):

    def test_new(self):

        class NewClass:
            pass
        new = NewClass()
        self.assertEqual(reflect.getClass(NewClass).__name__, 'type')
        self.assertEqual(reflect.getClass(new).__name__, 'NewClass')