import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
def test_processing_instruction_no_data_2(self):
    text = u'<?experiment>...<?/experiment>'
    events = list(HTMLParser(StringIO(text)))
    kind, (target, data), pos = events[0]
    self.assertEqual(Stream.PI, kind)
    self.assertEqual('experiment', target)
    self.assertEqual('', data)
    kind, (target, data), pos = events[2]
    self.assertEqual('/experiment', target)
    self.assertEqual('', data)