from __future__ import annotations
import errno
import socket
import sys
from typing import Sequence
from twisted.internet import error
from twisted.trial import unittest
class SubclassingTests(unittest.SynchronousTestCase):
    """
    Some exceptions are subclasses of other exceptions.
    """

    def test_connectionLostSubclassOfConnectionClosed(self) -> None:
        """
        L{error.ConnectionClosed} is a superclass of L{error.ConnectionLost}.
        """
        self.assertTrue(issubclass(error.ConnectionLost, error.ConnectionClosed))

    def test_connectionDoneSubclassOfConnectionClosed(self) -> None:
        """
        L{error.ConnectionClosed} is a superclass of L{error.ConnectionDone}.
        """
        self.assertTrue(issubclass(error.ConnectionDone, error.ConnectionClosed))

    def test_invalidAddressErrorSubclassOfValueError(self) -> None:
        """
        L{ValueError} is a superclass of L{error.InvalidAddressError}.
        """
        self.assertTrue(issubclass(error.InvalidAddressError, ValueError))