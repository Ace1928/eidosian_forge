from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_small_fragments(self):
    pktlines = []
    parser = PktLineParser(pktlines.append)
    parser.parse(b'00')
    parser.parse(b'05')
    parser.parse(b'z0000')
    self.assertEqual(pktlines, [b'z', None])
    self.assertEqual(b'', parser.get_tail())