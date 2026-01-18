import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_undocumented_property(self):

    class A(object):

        @misc.cachedproperty
        def b(self):
            return 'b'
    self.assertIsNone(inspect.getdoc(A.b))