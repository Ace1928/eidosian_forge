from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_write_across_boundary(self):
    self._writer.write(b'foo')
    self._writer.write(b'barbaz')
    self.assertOutputEquals(b'0007foo000abarba')
    self._truncate()
    self._writer.flush()
    self.assertOutputEquals(b'z')