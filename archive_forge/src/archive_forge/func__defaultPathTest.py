import os
import signal
import stat
import sys
import warnings
from unittest import skipIf
from twisted.internet import error, interfaces, reactor, utils
from twisted.internet.defer import Deferred
from twisted.python.runtime import platform
from twisted.python.test.test_util import SuppressedWarningsTests
from twisted.trial.unittest import SynchronousTestCase, TestCase
def _defaultPathTest(self, utilFunc, check):
    dir = os.path.abspath(self.mktemp())
    os.makedirs(dir)
    scriptFile = self.makeSourceFile(['import os, sys', 'cdir = os.getcwd()', 'sys.stdout.write(cdir)'])
    self.addCleanup(os.chdir, os.getcwd())
    os.chdir(dir)
    originalMode = stat.S_IMODE(os.stat('.').st_mode)
    os.chmod(dir, stat.S_IXUSR | stat.S_IRUSR)
    self.addCleanup(os.chmod, dir, originalMode)
    d = utilFunc(self.exe, ['-u', scriptFile])
    d.addCallback(check, dir.encode(sys.getfilesystemencoding()))
    return d