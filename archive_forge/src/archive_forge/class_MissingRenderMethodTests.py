from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
class MissingRenderMethodTests(unittest.TestCase):
    """
    Tests for how L{MissingRenderMethod} exceptions are initialized and
    displayed.
    """

    def test_constructor(self) -> None:
        """
        Given C{element} and C{renderName} arguments, the
        L{MissingRenderMethod} constructor assigns the values to the
        corresponding attributes.
        """
        elt = object()
        e = error.MissingRenderMethod(elt, 'renderThing')
        self.assertIs(e.element, elt)
        self.assertIs(e.renderName, 'renderThing')

    def test_repr(self) -> None:
        """
        A L{MissingRenderMethod} is represented using a custom string
        containing the element's representation and the method name.
        """
        elt = object()
        e = error.MissingRenderMethod(elt, 'renderThing')
        self.assertEqual(repr(e), "'MissingRenderMethod': %r had no render method named 'renderThing'" % elt)