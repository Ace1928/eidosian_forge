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
def _pathTest(self, utilFunc, check):
    dir = os.path.abspath(self.mktemp())
    os.makedirs(dir)
    scriptFile = self.makeSourceFile(['import os, sys', 'sys.stdout.write(os.getcwd())'])
    d = utilFunc(self.exe, ['-u', scriptFile], path=dir)
    d.addCallback(check, dir.encode(sys.getfilesystemencoding()))
    return d