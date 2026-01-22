import os
import pdb
import sys
import unittest as pyunit
from io import StringIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import plugin
from twisted.internet import defer
from twisted.plugins import twisted_trial
from twisted.python import failure, log, reflect
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.scripts import trial
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator
from twisted.trial.itrial import IReporter, ITestCase
class TrialRunnerWithUncleanWarningsReporterTests(TrialRunnerTestsMixin, unittest.SynchronousTestCase):
    """
    Tests for the TrialRunner's interaction with an unclean-error suppressing
    reporter.
    """

    def setUp(self):
        self.stream = StringIO()
        self.runner = runner.TrialRunner(CapturingReporter, stream=self.stream, uncleanWarnings=True)
        self.test = TrialRunnerTests('test_empty')