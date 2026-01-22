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
class ConnectionNotifyServerFactory(pb.PBServerFactory):
    """
    A server factory which stores the last connection and fires a
    L{Deferred} on connection made. This factory can handle only one
    client connection.

    @ivar protocolInstance: the last protocol instance.
    @type protocolInstance: C{pb.Broker}

    @ivar connectionMade: the deferred fired upon connection.
    @type connectionMade: C{Deferred}
    """
    protocolInstance = None

    def __init__(self, root):
        """
        Initialize the factory.
        """
        pb.PBServerFactory.__init__(self, root)
        self.connectionMade = Deferred()

    def clientConnectionMade(self, protocol):
        """
        Store the protocol and fire the connection deferred.
        """
        self.protocolInstance = protocol
        d, self.connectionMade = (self.connectionMade, None)
        if d is not None:
            d.callback(None)