import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
@implementer(IForwarded)
class Forwarded:
    """
    Test implementation of L{IForwarded}.

    @ivar forwarded: set if C{forwardMe} is called.
    @type forwarded: C{bool}
    @ivar unforwarded: set if C{dontForwardMe} is called.
    @type unforwarded: C{bool}
    """
    forwarded = False
    unforwarded = False

    def forwardMe(self):
        """
        Set a local flag to test afterwards.
        """
        self.forwarded = True

    def dontForwardMe(self):
        """
        Set a local flag to test afterwards. This should not be called as it's
        not in the interface.
        """
        self.unforwarded = True

    def forwardDeferred(self):
        """
        Asynchronously return C{True}.
        """
        return succeed(True)