import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_text_node_pos_single_line(self):
    text = u'<elem>foo bar</elem>'
    events = list(HTMLParser(StringIO(text)))
    kind, data, pos = events[1]
    self.assertEqual(Stream.TEXT, kind)
    self.assertEqual('foo bar', data)
    self.assertEqual((None, 1, 6), pos)