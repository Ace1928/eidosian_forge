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
def test_walksDownPath(self):
    """
        C{_qualNameWalker} is a generator that, when given a Python qualified
        name, yields that name, and then the parent of that name, and so forth,
        along with a list of the tried components, in a 2-tuple.
        """
    walkerResults = list(runner._qualNameWalker('walker.texas.ranger'))
    self.assertEqual(walkerResults, [('walker.texas.ranger', []), ('walker.texas', ['ranger']), ('walker', ['texas', 'ranger'])])