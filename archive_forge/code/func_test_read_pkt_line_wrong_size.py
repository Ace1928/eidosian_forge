from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_read_pkt_line_wrong_size(self):
    self.rin.write(b'0100too short')
    self.rin.seek(0)
    self.assertRaises(GitProtocolError, self.proto.read_pkt_line)