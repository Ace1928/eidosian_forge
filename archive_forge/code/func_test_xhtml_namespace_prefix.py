import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_xhtml_namespace_prefix(self):
    text = '<div xmlns="http://www.w3.org/1999/xhtml">\n            <strong>Hello</strong>\n        </div>'
    output = XML(text).render(XHTMLSerializer, encoding=None)
    self.assertEqual(text, output)