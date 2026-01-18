import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_xmldecl(self):
    text = '<?xml version="1.0" ?><root />'
    events = list(XMLParser(StringIO(text)))
    kind, (version, encoding, standalone), pos = events[0]
    self.assertEqual(Stream.XML_DECL, kind)
    self.assertEqual('1.0', version)
    self.assertEqual(None, encoding)
    self.assertEqual(-1, standalone)