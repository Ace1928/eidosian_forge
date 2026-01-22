import gc
import weakref
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
class object_with_finalizer(object):

    def __del__(self):
        pass