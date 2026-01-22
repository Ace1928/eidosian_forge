import gc
import weakref
import greenlet
from . import TestCase
def _dead_greenlet():
    g = greenlet.greenlet(lambda: None)
    g.switch()
    return g