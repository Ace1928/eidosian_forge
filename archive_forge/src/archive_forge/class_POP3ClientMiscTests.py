import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
class POP3ClientMiscTests(TestCase):

    def testCapability(self):
        p, t = setUp()
        d = p.capabilities(useCache=0)
        self.assertEqual(t.value(), b'CAPA\r\n')
        p.dataReceived(b'+OK Capabilities on the way\r\n')
        p.dataReceived(b'X\r\nY\r\nZ\r\nA 1 2 3\r\nB 1 2\r\nC 1\r\n.\r\n')
        return d.addCallback(self.assertEqual, {b'X': None, b'Y': None, b'Z': None, b'A': [b'1', b'2', b'3'], b'B': [b'1', b'2'], b'C': [b'1']})

    def testCapabilityError(self):
        p, t = setUp()
        d = p.capabilities(useCache=0)
        self.assertEqual(t.value(), b'CAPA\r\n')
        p.dataReceived(b'-ERR This server is lame!\r\n')
        return d.addCallback(self.assertEqual, {})

    def testStat(self):
        p, t = setUp()
        d = p.stat()
        self.assertEqual(t.value(), b'STAT\r\n')
        p.dataReceived(b'+OK 1 1212\r\n')
        return d.addCallback(self.assertEqual, (1, 1212))

    def testStatError(self):
        p, t = setUp()
        d = p.stat()
        self.assertEqual(t.value(), b'STAT\r\n')
        p.dataReceived(b'-ERR This server is lame!\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'This server is lame!'))

    def testNoop(self):
        p, t = setUp()
        d = p.noop()
        self.assertEqual(t.value(), b'NOOP\r\n')
        p.dataReceived(b'+OK No-op to you too!\r\n')
        return d.addCallback(self.assertEqual, b'No-op to you too!')

    def testNoopError(self):
        p, t = setUp()
        d = p.noop()
        self.assertEqual(t.value(), b'NOOP\r\n')
        p.dataReceived(b'-ERR This server is lame!\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'This server is lame!'))

    def testRset(self):
        p, t = setUp()
        d = p.reset()
        self.assertEqual(t.value(), b'RSET\r\n')
        p.dataReceived(b'+OK Reset state\r\n')
        return d.addCallback(self.assertEqual, b'Reset state')

    def testRsetError(self):
        p, t = setUp()
        d = p.reset()
        self.assertEqual(t.value(), b'RSET\r\n')
        p.dataReceived(b'-ERR This server is lame!\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'This server is lame!'))

    def testDelete(self):
        p, t = setUp()
        d = p.delete(3)
        self.assertEqual(t.value(), b'DELE 4\r\n')
        p.dataReceived(b'+OK Hasta la vista\r\n')
        return d.addCallback(self.assertEqual, b'Hasta la vista')

    def testDeleteError(self):
        p, t = setUp()
        d = p.delete(3)
        self.assertEqual(t.value(), b'DELE 4\r\n')
        p.dataReceived(b'-ERR Winner is not you.\r\n')
        return self.assertFailure(d, ServerErrorResponse).addCallback(lambda exc: self.assertEqual(exc.args[0], b'Winner is not you.'))