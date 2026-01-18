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
def test_whichPathExt(self):
    j = os.path.join
    old = os.environ.get('PATHEXT', None)
    os.environ['PATHEXT'] = os.pathsep.join(('.bin', '.exe', '.sh'))
    try:
        paths = procutils.which('executable')
    finally:
        if old is None:
            del os.environ['PATHEXT']
        else:
            os.environ['PATHEXT'] = old
    expectedPaths = [j(self.foobaz, 'executable'), j(self.bazfoo, 'executable'), j(self.bazfoo, 'executable.bin')]
    if runtime.platform.isWindows():
        expectedPaths.append(j(self.bazbar, 'executable'))
    self.assertEqual(paths, expectedPaths)