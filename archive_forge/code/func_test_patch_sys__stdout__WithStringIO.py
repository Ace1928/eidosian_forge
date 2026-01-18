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
def test_patch_sys__stdout__WithStringIO(self):
    """
        If L{sys.stdout} and L{sys.__stdout__} are patched with L{io.StringIO},
        we should get a L{ValueError}.
        """
    import sys
    from io import StringIO
    self.patch(sys, 'stdout', StringIO())
    self.patch(sys, '__stdout__', StringIO())
    return self.test_stdio()