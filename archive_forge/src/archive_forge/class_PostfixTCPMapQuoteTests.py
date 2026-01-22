from typing import Dict, List, Tuple
from twisted.internet.testing import StringTransport
from twisted.protocols import postfix
from twisted.trial import unittest
class PostfixTCPMapQuoteTests(unittest.TestCase):
    data = [(b'foo', b'foo'), (b'foo bar', b'foo%20bar'), (b'foo\tbar', b'foo%09bar'), (b'foo\nbar', b'foo%0Abar', b'foo%0abar'), (b'foo\r\nbar', b'foo%0D%0Abar', b'foo%0D%0abar', b'foo%0d%0Abar', b'foo%0d%0abar'), (b'foo ', b'foo%20'), (b' foo', b'%20foo')]

    def testData(self):
        for entry in self.data:
            raw = entry[0]
            quoted = entry[1:]
            self.assertEqual(postfix.quote(raw), quoted[0])
            for q in quoted:
                self.assertEqual(postfix.unquote(q), raw)