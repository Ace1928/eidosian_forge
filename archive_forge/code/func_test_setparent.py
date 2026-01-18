from __future__ import print_function
from __future__ import absolute_import
import sys
import greenlet
from . import _test_extension
from . import TestCase
def test_setparent(self):

    def foo():

        def bar():
            greenlet.getcurrent().parent.switch()
            greenlet.getcurrent().parent.switch()
            raise AssertionError('Should never have reached this code')
        child = greenlet.greenlet(bar)
        child.switch()
        greenlet.getcurrent().parent.switch(child)
        greenlet.getcurrent().parent.throw(AssertionError('Should never reach this code'))
    foo_child = greenlet.greenlet(foo).switch()
    self.assertEqual(None, _test_extension.test_setparent(foo_child))