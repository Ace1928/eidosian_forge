import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
class UntilConcludesTests(TestCase):
    """
    Tests for L{untilConcludes}, an C{EINTR} helper.
    """

    def test_uninterruptably(self):
        """
        L{untilConcludes} calls the function passed to it until the function
        does not raise either L{OSError} or L{IOError} with C{errno} of
        C{EINTR}.  It otherwise completes with the same result as the function
        passed to it.
        """

        def f(a, b):
            self.calls += 1
            exc = self.exceptions.pop()
            if exc is not None:
                raise exc(errno.EINTR, 'Interrupted system call!')
            return a + b
        self.exceptions = [None]
        self.calls = 0
        self.assertEqual(util.untilConcludes(f, 1, 2), 3)
        self.assertEqual(self.calls, 1)
        self.exceptions = [None, OSError, IOError]
        self.calls = 0
        self.assertEqual(util.untilConcludes(f, 2, 3), 5)
        self.assertEqual(self.calls, 3)