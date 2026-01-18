from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_write_multiple(self):
    self._writer.write(b'foo')
    self._writer.write(b'bar')
    self.assertOutputEquals(b'')
    self._writer.flush()
    self.assertOutputEquals(b'0007foo0007bar')