from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def assertOutputEquals(self, expected):
    self.assertEqual(expected, self._output.getvalue())