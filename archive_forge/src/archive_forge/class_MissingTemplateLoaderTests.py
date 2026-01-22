from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
class MissingTemplateLoaderTests(unittest.TestCase):
    """
    Tests for how L{MissingTemplateLoader} exceptions are initialized and
    displayed.
    """

    def test_constructor(self) -> None:
        """
        Given an C{element} argument, the L{MissingTemplateLoader} constructor
        assigns the value to the corresponding attribute.
        """
        elt = object()
        e = error.MissingTemplateLoader(elt)
        self.assertIs(e.element, elt)

    def test_repr(self) -> None:
        """
        A L{MissingTemplateLoader} is represented using a custom string
        containing the element's representation and the method name.
        """
        elt = object()
        e = error.MissingTemplateLoader(elt)
        self.assertEqual(repr(e), "'MissingTemplateLoader': %r had no loader" % elt)