import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class SafeReprTests(TestCase):
    """
    Tests for L{reflect.safe_repr} function.
    """

    def test_workingRepr(self):
        """
        L{reflect.safe_repr} produces the same output as C{repr} on a working
        object.
        """
        xs = ([1, 2, 3], b'a')
        self.assertEqual(list(map(reflect.safe_repr, xs)), list(map(repr, xs)))

    def test_brokenRepr(self):
        """
        L{reflect.safe_repr} returns a string with class name, address, and
        traceback when the repr call failed.
        """
        b = Breakable()
        b.breakRepr = True
        bRepr = reflect.safe_repr(b)
        self.assertIn('Breakable instance at 0x', bRepr)
        self.assertIn(os.path.splitext(__file__)[0], bRepr)
        self.assertIn('RuntimeError: repr!', bRepr)

    def test_brokenStr(self):
        """
        L{reflect.safe_repr} isn't affected by a broken C{__str__} method.
        """
        b = Breakable()
        b.breakStr = True
        self.assertEqual(reflect.safe_repr(b), repr(b))

    def test_brokenClassRepr(self):

        class X(BTBase):
            breakRepr = True
        reflect.safe_repr(X)
        reflect.safe_repr(X())

    def test_brokenReprIncludesID(self):
        """
        C{id} is used to print the ID of the object in case of an error.

        L{safe_repr} includes a traceback after a newline, so we only check
        against the first line of the repr.
        """

        class X(BTBase):
            breakRepr = True
        xRepr = reflect.safe_repr(X)
        xReprExpected = f'<BrokenType instance at 0x{id(X):x} with repr error:'
        self.assertEqual(xReprExpected, xRepr.split('\n')[0])

    def test_brokenClassStr(self):

        class X(BTBase):
            breakStr = True
        reflect.safe_repr(X)
        reflect.safe_repr(X())

    def test_brokenClassAttribute(self):
        """
        If an object raises an exception when accessing its C{__class__}
        attribute, L{reflect.safe_repr} uses C{type} to retrieve the class
        object.
        """
        b = NoClassAttr()
        b.breakRepr = True
        bRepr = reflect.safe_repr(b)
        self.assertIn('NoClassAttr instance at 0x', bRepr)
        self.assertIn(os.path.splitext(__file__)[0], bRepr)
        self.assertIn('RuntimeError: repr!', bRepr)

    def test_brokenClassNameAttribute(self):
        """
        If a class raises an exception when accessing its C{__name__} attribute
        B{and} when calling its C{__str__} implementation, L{reflect.safe_repr}
        returns 'BROKEN CLASS' instead of the class name.
        """

        class X(BTBase):
            breakName = True
        xRepr = reflect.safe_repr(X())
        self.assertIn('<BROKEN CLASS AT 0x', xRepr)
        self.assertIn(os.path.splitext(__file__)[0], xRepr)
        self.assertIn('RuntimeError: repr!', xRepr)