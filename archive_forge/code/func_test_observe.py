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
def test_observe(self):
    c, s, pump = connectedServerAndClient(test=self)
    a = Observable()
    b = Observer()
    s.setNameForLocal('a', a)
    ra = c.remoteForName('a')
    ra.callRemote('observe', b)
    pump.pump()
    a.notify(1)
    pump.pump()
    pump.pump()
    a.notify(10)
    pump.pump()
    pump.pump()
    self.assertIsNotNone(b.obj, "didn't notify")
    self.assertEqual(b.obj, 1, 'notified too much')