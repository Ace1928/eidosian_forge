import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
def test_xml_lang(self):
    text = '<p xml:lang="en">English text</p>'
    output = XML(text).render(HTMLSerializer, encoding=None)
    self.assertEqual('<p lang="en">English text</p>', output)