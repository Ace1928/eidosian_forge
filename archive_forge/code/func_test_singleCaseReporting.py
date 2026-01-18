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
def test_singleCaseReporting(self):
    """
        If we are running a single test, check the reporter starts, passes and
        then stops the test during a dry run.
        """
    result = self.runner.run(self.test)
    self.assertEqual(result._calls, ['startTest', 'addSuccess', 'stopTest'])