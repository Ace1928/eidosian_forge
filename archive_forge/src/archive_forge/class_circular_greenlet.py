import gc
import weakref
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
class circular_greenlet(greenlet.greenlet):
    self = None