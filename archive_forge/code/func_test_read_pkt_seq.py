from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_read_pkt_seq(self):
    self.rin.write(b'0008cmd 0005l0000')
    self.rin.seek(0)
    self.assertEqual([b'cmd ', b'l'], list(self.proto.read_pkt_seq()))