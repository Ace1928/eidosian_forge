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
class _ReconnectingFakeConnector(_FakeConnector):
    """
    A fake L{IConnector} that can fire L{Deferred}s when its
    C{connect} method is called.
    """

    def __init__(self, address, state):
        """
        @param address: An L{IAddress} provider that represents this
            connector's destination.
        @type address: An L{IAddress} provider.

        @param state: The state instance
        @type state: L{_ReconnectingFakeConnectorState}
        """
        super().__init__(address)
        self._state = state

    def connect(self):
        """
        A C{connect} implementation that calls C{reconnectCallback}
        """
        super().connect()
        self._state.notifyAll()