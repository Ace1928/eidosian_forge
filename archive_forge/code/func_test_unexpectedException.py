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
def test_unexpectedException(self):
    """
        When the checker specified by C{--auth} raises an unexpected error, it
        should be caught and re-raised within a L{usage.UsageError}.
        """
    options = DummyOptions()
    err = self.assertRaises(usage.UsageError, options.parseOptions, ['--auth', 'file'])
    self.assertEqual(str(err), "Unexpected error: 'file' requires a filename")