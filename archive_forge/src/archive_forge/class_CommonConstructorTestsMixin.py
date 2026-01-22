import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class CommonConstructorTestsMixin:
    """
    Tests for constructor arguments and their associated attributes that are
    common to both L{twisted.names.dns._EDNSMessage} and L{dns.Message}.

    TestCase classes that use this mixin must provide a C{messageFactory} method
    which accepts any argment supported by L{dns.Message.__init__}.

    TestCases must also mixin ConstructorTestsMixin which provides some custom
    assertions for testing constructor arguments.
    """

    def test_id(self):
        """
        L{dns._EDNSMessage.id} defaults to C{0} and can be overridden in
        the constructor.
        """
        self._verifyConstructorArgument('id', defaultVal=0, altVal=1)

    def test_answer(self):
        """
        L{dns._EDNSMessage.answer} defaults to C{False} and can be overridden in
        the constructor.
        """
        self._verifyConstructorFlag('answer', defaultVal=False)

    def test_opCode(self):
        """
        L{dns._EDNSMessage.opCode} defaults to L{dns.OP_QUERY} and can be
        overridden in the constructor.
        """
        self._verifyConstructorArgument('opCode', defaultVal=dns.OP_QUERY, altVal=dns.OP_STATUS)

    def test_auth(self):
        """
        L{dns._EDNSMessage.auth} defaults to C{False} and can be overridden in
        the constructor.
        """
        self._verifyConstructorFlag('auth', defaultVal=False)

    def test_trunc(self):
        """
        L{dns._EDNSMessage.trunc} defaults to C{False} and can be overridden in
        the constructor.
        """
        self._verifyConstructorFlag('trunc', defaultVal=False)

    def test_recDes(self):
        """
        L{dns._EDNSMessage.recDes} defaults to C{False} and can be overridden in
        the constructor.
        """
        self._verifyConstructorFlag('recDes', defaultVal=False)

    def test_recAv(self):
        """
        L{dns._EDNSMessage.recAv} defaults to C{False} and can be overridden in
        the constructor.
        """
        self._verifyConstructorFlag('recAv', defaultVal=False)

    def test_rCode(self):
        """
        L{dns._EDNSMessage.rCode} defaults to C{0} and can be overridden in the
        constructor.
        """
        self._verifyConstructorArgument('rCode', defaultVal=0, altVal=123)

    def test_maxSize(self):
        """
        L{dns._EDNSMessage.maxSize} defaults to C{512} and can be overridden in
        the constructor.
        """
        self._verifyConstructorArgument('maxSize', defaultVal=512, altVal=1024)

    def test_queries(self):
        """
        L{dns._EDNSMessage.queries} defaults to C{[]}.
        """
        self.assertEqual(self.messageFactory().queries, [])

    def test_answers(self):
        """
        L{dns._EDNSMessage.answers} defaults to C{[]}.
        """
        self.assertEqual(self.messageFactory().answers, [])

    def test_authority(self):
        """
        L{dns._EDNSMessage.authority} defaults to C{[]}.
        """
        self.assertEqual(self.messageFactory().authority, [])

    def test_additional(self):
        """
        L{dns._EDNSMessage.additional} defaults to C{[]}.
        """
        self.assertEqual(self.messageFactory().additional, [])