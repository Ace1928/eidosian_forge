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
class LeakyRealm(TestRealm):
    """
    A realm which hangs onto a reference to the mind object in its logout
    function.
    """

    def __init__(self, mindEater):
        """
        Create a L{LeakyRealm}.

        @param mindEater: a callable that will be called with the C{mind}
        object when it is available
        """
        self._mindEater = mindEater

    def requestAvatar(self, avatarId, mind, interface):
        self._mindEater(mind)
        persp = self.perspectiveFactory(avatarId)
        return (pb.IPerspective, persp, lambda: (mind, persp.logout()))