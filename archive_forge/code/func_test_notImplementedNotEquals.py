import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_notImplementedNotEquals(self):
    """
        Instances of a class that is decorated by C{comparable} support
        returning C{NotImplemented} from C{__ne__} if it is returned by the
        underlying C{__cmp__} call.
        """
    self.assertEqual(Comparable(1).__ne__(object()), NotImplemented)