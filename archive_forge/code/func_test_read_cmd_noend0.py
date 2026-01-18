from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_read_cmd_noend0(self):
    self.rin.write(b'0011cmd arg1\x00arg2')
    self.rin.seek(0)
    self.assertRaises(AssertionError, self.proto.read_cmd)