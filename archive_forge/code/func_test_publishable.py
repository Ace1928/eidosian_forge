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
def test_publishable(self):
    try:
        os.unlink('None-None-TESTING.pub')
    except OSError:
        pass
    c, s, pump = connectedServerAndClient(test=self)
    foo = GetPublisher()
    s.setNameForLocal('foo', foo)
    bar = c.remoteForName('foo')
    accum = []
    bar.callRemote('getPub').addCallbacks(accum.append, self.thunkErrorBad)
    pump.flush()
    obj = accum.pop()
    self.assertEqual(obj.activateCalled, 1)
    self.assertEqual(obj.isActivated, 1)
    self.assertEqual(obj.yayIGotPublished, 1)
    self.assertEqual(obj._wasCleanWhenLoaded, 0)
    c, s, pump = connectedServerAndClient(test=self)
    s.setNameForLocal('foo', foo)
    bar = c.remoteForName('foo')
    bar.callRemote('getPub').addCallbacks(accum.append, self.thunkErrorBad)
    pump.flush()
    obj = accum.pop()
    self.assertEqual(obj._wasCleanWhenLoaded, 1)