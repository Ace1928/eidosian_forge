from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
@skipIf(not isGraphvizModuleInstalled(), 'Graphviz module is not installed.')
@skipIf(not isTwistedInstalled(), 'Twisted is not installed.')
class ElementMakerTests(TestCase):
    """
    L{elementMaker} generates HTML representing the specified element.
    """

    def setUp(self):
        from .._visualize import elementMaker
        self.elementMaker = elementMaker

    def test_sortsAttrs(self):
        """
        L{elementMaker} orders HTML attributes lexicographically.
        """
        expected = '<div a="1" b="2" c="3"></div>'
        self.assertEqual(expected, self.elementMaker('div', b='2', a='1', c='3'))

    def test_quotesAttrs(self):
        """
        L{elementMaker} quotes HTML attributes according to DOT's quoting rule.

        See U{http://www.graphviz.org/doc/info/lang.html}, footnote 1.
        """
        expected = '<div a="1" b="a \\" quote" c="a string"></div>'
        self.assertEqual(expected, self.elementMaker('div', b='a " quote', a=1, c='a string'))

    def test_noAttrs(self):
        """
        L{elementMaker} should render an element with no attributes.
        """
        expected = '<div ></div>'
        self.assertEqual(expected, self.elementMaker('div'))