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
def test_stdinReader_unicodeArgs(self):
    """
        Pass L{unicode} args to L{_test_stdinReader}.
        """
    import win32api
    pyExe = FilePath(sys.executable).path
    args = [pyExe, '-u', '-m', 'twisted.test.process_stdinreader']
    env = properEnv
    pythonPath = os.pathsep.join(sys.path)
    env['PYTHONPATH'] = pythonPath
    path = win32api.GetTempPath()
    d = self._test_stdinReader(pyExe, args, env, path)
    return d