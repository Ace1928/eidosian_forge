import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_stringToString(self):
    """
        C{nativeString} leaves native strings as native strings.
        """
    self.assertNativeString('Hello!', 'Hello!')