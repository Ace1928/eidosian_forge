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
def test_suppressWarnings(self):
    """
        L{utils.suppressWarnings} decorates a function so that the given
        warnings are suppressed.
        """
    result = []

    def showwarning(self, *a, **kw):
        result.append((a, kw))
    self.patch(warnings, 'showwarning', showwarning)

    def f(msg):
        warnings.warn(msg)
    g = utils.suppressWarnings(f, (('ignore',), dict(message='This is message')))
    f('Sanity check message')
    self.assertEqual(len(result), 1)
    g('This is message')
    self.assertEqual(len(result), 1)
    g('Unignored message')
    self.assertEqual(len(result), 2)