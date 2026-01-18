from __future__ import print_function
from __future__ import absolute_import
import sys
import greenlet
from . import _test_extension
from . import TestCase
def test_throw(self):
    seen = []

    def foo():
        try:
            greenlet.getcurrent().parent.switch()
        except ValueError:
            seen.append(sys.exc_info()[1])
        except greenlet.GreenletExit:
            raise AssertionError
    g = greenlet.greenlet(foo)
    g.switch()
    _test_extension.test_throw(g)
    self.assertEqual(len(seen), 1)
    self.assertTrue(isinstance(seen[0], ValueError), 'ValueError was not raised in foo()')
    self.assertEqual(str(seen[0]), 'take that sucka!', "message doesn't match")