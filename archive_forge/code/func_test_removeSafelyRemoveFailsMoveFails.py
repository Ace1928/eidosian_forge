from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
def test_removeSafelyRemoveFailsMoveFails(self) -> None:
    """
        If an L{OSError} is raised while removing a path in
        L{util._removeSafely}, an attempt is made to move the path to a new
        name. If that attempt fails, the L{OSError} is re-raised.
        """

    def dummyRemove() -> None:
        """
            Raise an C{OSError} to emulate the branch of L{util._removeSafely}
            in which path removal fails.
            """
        raise OSError('path removal failed')

    def dummyMoveTo(destination: object, followLinks: bool=True) -> None:
        """
            Raise an C{OSError} to emulate the branch of L{util._removeSafely}
            in which path movement fails.
            """
        raise OSError('path movement failed')
    out = StringIO()
    self.patch(sys, 'stdout', out)
    directory = self.mktemp().encode('utf-8')
    os.mkdir(directory)
    dirPath = filepath.FilePath(directory)
    dirPath.child(b'_trial_marker').touch()
    dirPath.remove = dummyRemove
    dirPath.moveTo = dummyMoveTo
    error = self.assertRaises(OSError, util._removeSafely, dirPath)
    self.assertEqual(str(error), 'path movement failed')
    self.assertIn('could not remove FilePath', out.getvalue())