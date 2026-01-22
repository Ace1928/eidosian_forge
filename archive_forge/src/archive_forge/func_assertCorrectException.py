from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
def assertCorrectException(self, errno: int | None, message: object, result: error.ConnectError, expectedClass: type[error.ConnectError]) -> None:
    """
        The given result of L{error.getConnectError} has the given attributes
        (C{osError} and C{args}), and is an instance of the given class.
        """
    self.assertEqual(result.__class__, expectedClass)
    self.assertEqual(result.osError, errno)
    self.assertEqual(result.args, (message,))