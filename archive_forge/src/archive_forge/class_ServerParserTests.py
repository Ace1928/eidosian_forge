import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
class ServerParserTests(unittest.TestCase):

    def testErrors(self):
        p = TestErrorIdentServer()
        p.makeConnection(StringTransport())
        L = []
        p.sendLine = L.append
        p.exceptionType = ident.IdentError
        p.lineReceived('123, 345')
        self.assertEqual(L[0], '123, 345 : ERROR : UNKNOWN-ERROR')
        p.exceptionType = ident.NoUser
        p.lineReceived('432, 210')
        self.assertEqual(L[1], '432, 210 : ERROR : NO-USER')
        p.exceptionType = ident.InvalidPort
        p.lineReceived('987, 654')
        self.assertEqual(L[2], '987, 654 : ERROR : INVALID-PORT')
        p.exceptionType = ident.HiddenUser
        p.lineReceived('756, 827')
        self.assertEqual(L[3], '756, 827 : ERROR : HIDDEN-USER')
        p.exceptionType = NewException
        p.lineReceived('987, 789')
        self.assertEqual(L[4], '987, 789 : ERROR : UNKNOWN-ERROR')
        errs = self.flushLoggedErrors(NewException)
        self.assertEqual(len(errs), 1)
        for port in (-1, 0, 65536, 65537):
            del L[:]
            p.lineReceived('%d, 5' % (port,))
            p.lineReceived('5, %d' % (port,))
            self.assertEqual(L, ['%d, 5 : ERROR : INVALID-PORT' % (port,), '5, %d : ERROR : INVALID-PORT' % (port,)])

    def testSuccess(self):
        p = TestIdentServer()
        p.makeConnection(StringTransport())
        L = []
        p.sendLine = L.append
        p.resultValue = ('SYS', 'USER')
        p.lineReceived('123, 456')
        self.assertEqual(L[0], '123, 456 : USERID : SYS : USER')