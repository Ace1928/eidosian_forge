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
class ExcInfoHoldingErrorHolderTests(ErrorHolderTestsMixin, TestHolderTests):
    """
    Tests for L{runner.ErrorHolder} behaving similarly to L{runner.TestHolder}
    when constructed with a C{exc_info}-style tuple representing its error.
    """

    def setUp(self):
        self.description = 'description'
        try:
            raise self.exceptionForTests
        except ZeroDivisionError:
            exceptionInfo = sys.exc_info()
            self.error = failure.Failure()
        self.holder = runner.ErrorHolder(self.description, exceptionInfo)
        self.result = self.TestResultStub()