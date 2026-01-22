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
class SpreadUtilTests(unittest.TestCase):
    """
    Tests for L{twisted.spread.util}.
    """

    def test_sync(self):
        """
        Call a synchronous method of a L{util.LocalAsRemote} object and check
        the result.
        """
        o = LocalRemoteTest()
        self.assertEqual(o.callRemote('add1', 2), 3)

    def test_async(self):
        """
        Call an asynchronous method of a L{util.LocalAsRemote} object and check
        the result.
        """
        o = LocalRemoteTest()
        o = LocalRemoteTest()
        d = o.callRemote('add', 2, y=4)
        self.assertIsInstance(d, Deferred)
        d.addCallback(self.assertEqual, 6)
        return d

    def test_asyncFail(self):
        """
        Test an asynchronous failure on a remote method call.
        """
        o = LocalRemoteTest()
        d = o.callRemote('fail')

        def eb(f):
            self.assertIsInstance(f, failure.Failure)
            f.trap(RuntimeError)
        d.addCallbacks(lambda res: self.fail('supposed to fail'), eb)
        return d

    def test_remoteMethod(self):
        """
        Test the C{remoteMethod} facility of L{util.LocalAsRemote}.
        """
        o = LocalRemoteTest()
        m = o.remoteMethod('add1')
        self.assertEqual(m(3), 4)

    def test_localAsyncForwarder(self):
        """
        Test a call to L{util.LocalAsyncForwarder} using L{Forwarded} local
        object.
        """
        f = Forwarded()
        lf = util.LocalAsyncForwarder(f, IForwarded)
        lf.callRemote('forwardMe')
        self.assertTrue(f.forwarded)
        lf.callRemote('dontForwardMe')
        self.assertFalse(f.unforwarded)
        rr = lf.callRemote('forwardDeferred')
        l = []
        rr.addCallback(l.append)
        self.assertEqual(l[0], 1)