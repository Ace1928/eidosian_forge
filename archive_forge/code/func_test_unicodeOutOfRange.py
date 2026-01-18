import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_unicodeOutOfRange(self):
    """
        L{networkString} raises L{UnicodeError} if passed a C{unicode} instance
        containing characters not encodable in ASCII.
        """
    self.assertRaises(UnicodeError, networkString, '☃')