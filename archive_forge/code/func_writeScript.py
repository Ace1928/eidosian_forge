import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def writeScript(self, content):
    """
        Write L{content} to a new temporary file, returning the L{FilePath}
        for the new file.
        """
    path = self.mktemp()
    with open(path, 'wb') as f:
        f.write(content.encode('ascii'))
    return FilePath(path.encode('utf-8'))