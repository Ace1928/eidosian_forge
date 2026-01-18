from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_write_sideband(self):
    self.proto.write_sideband(3, b'bloe')
    self.assertEqual(self.rout.getvalue(), b'0009\x03bloe')