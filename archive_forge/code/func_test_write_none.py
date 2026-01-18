from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_write_none(self):
    self._writer.write(None)
    self.assertOutputEquals(b'')
    self._writer.flush()
    self.assertOutputEquals(b'0000')