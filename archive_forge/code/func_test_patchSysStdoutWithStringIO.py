import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
def test_patchSysStdoutWithStringIO(self):
    """
        Some projects which use the Twisted reactor
        such as Buildbot patch L{sys.stdout} with L{io.StringIO}
        before running their tests.
        """
    import sys
    from io import StringIO
    stdoutStringIO = StringIO()
    self.patch(sys, 'stdout', stdoutStringIO)
    return self.test_stdio()