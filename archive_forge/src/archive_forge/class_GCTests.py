import gc
import weakref
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
class GCTests(TestCase):

    def test_dead_circular_ref(self):
        o = weakref.ref(greenlet.greenlet(greenlet.getcurrent).switch())
        gc.collect()
        if o() is not None:
            import sys
            print('O IS NOT NONE.', sys.getrefcount(o()))
        self.assertIsNone(o())
        self.assertFalse(gc.garbage, gc.garbage)

    def test_circular_greenlet(self):

        class circular_greenlet(greenlet.greenlet):
            self = None
        o = circular_greenlet()
        o.self = o
        o = weakref.ref(o)
        gc.collect()
        self.assertIsNone(o())
        self.assertFalse(gc.garbage, gc.garbage)

    def test_inactive_ref(self):

        class inactive_greenlet(greenlet.greenlet):

            def __init__(self):
                greenlet.greenlet.__init__(self, run=self.run)

            def run(self):
                pass
        o = inactive_greenlet()
        o = weakref.ref(o)
        gc.collect()
        self.assertIsNone(o())
        self.assertFalse(gc.garbage, gc.garbage)

    @fails_leakcheck
    def test_finalizer_crash(self):

        class object_with_finalizer(object):

            def __del__(self):
                pass
        array = []
        parent = greenlet.getcurrent()

        def greenlet_body():
            greenlet.getcurrent().object = object_with_finalizer()
            try:
                parent.switch()
            except greenlet.GreenletExit:
                print('Got greenlet exit!')
            finally:
                del greenlet.getcurrent().object
        g = greenlet.greenlet(greenlet_body)
        g.array = array
        array.append(g)
        g.switch()
        del array
        del g
        greenlet.getcurrent()
        gc.collect()