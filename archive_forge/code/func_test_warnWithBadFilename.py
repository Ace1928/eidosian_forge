import os
from io import StringIO
from typing import Sequence, Type
from unittest import skipIf
from zope.interface import Interface
from twisted import plugin
from twisted.cred import checkers, credentials, error, strcred
from twisted.plugins import cred_anonymous, cred_file, cred_unix
from twisted.python import usage
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_warnWithBadFilename(self):
    """
        When the file auth plugin is given a file that doesn't exist, it
        should produce a warning.
        """
    oldOutput = cred_file.theFileCheckerFactory.errorOutput
    newOutput = StringIO()
    cred_file.theFileCheckerFactory.errorOutput = newOutput
    strcred.makeChecker('file:' + self._fakeFilename())
    cred_file.theFileCheckerFactory.errorOutput = oldOutput
    self.assertIn(cred_file.invalidFileWarning, newOutput.getvalue())