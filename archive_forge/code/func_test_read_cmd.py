from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_read_cmd(self):
    self.rin.write(b'0012cmd arg1\x00arg2\x00')
    self.rin.seek(0)
    self.assertEqual((b'cmd', [b'arg1', b'arg2']), self.proto.read_cmd())