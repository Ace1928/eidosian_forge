import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_style_escaping(self):
    text = '<style>html &gt; body { display: none; }</style>'
    output = XML(text).render(HTMLSerializer, encoding=None)
    self.assertEqual('<style>html > body { display: none; }</style>', output)