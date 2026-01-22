import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class IReadWriteTestsMixin:
    """
    Generic tests for the C{IReadFile} and C{IWriteFile} interfaces.
    """

    def getFileReader(self, content):
        """
        Return an object providing C{IReadFile}, ready to send data C{content}.

        @param content: data to send
        """
        raise NotImplementedError()

    def getFileWriter(self):
        """
        Return an object providing C{IWriteFile}, ready to receive data.
        """
        raise NotImplementedError()

    def getFileContent(self):
        """
        Return the content of the file used.
        """
        raise NotImplementedError()

    def test_read(self):
        """
        Test L{ftp.IReadFile}: the implementation should have a send method
        returning a C{Deferred} which fires when all the data has been sent
        to the consumer, and the data should be correctly send to the consumer.
        """
        content = b'wobble\n'
        consumer = TestConsumer()

        def cbGet(reader):
            return reader.send(consumer).addCallback(cbSend)

        def cbSend(res):
            self.assertEqual(b''.join(consumer.buffer), content)
        return self.getFileReader(content).addCallback(cbGet)

    def test_write(self):
        """
        Test L{ftp.IWriteFile}: the implementation should have a receive
        method returning a C{Deferred} which fires with a consumer ready to
        receive data to be written. It should also have a close() method that
        returns a Deferred.
        """
        content = b'elbbow\n'

        def cbGet(writer):
            return writer.receive().addCallback(cbReceive, writer)

        def cbReceive(consumer, writer):
            producer = TestProducer(content, consumer)
            consumer.registerProducer(None, True)
            producer.start()
            consumer.unregisterProducer()
            return writer.close().addCallback(cbClose)

        def cbClose(ignored):
            self.assertEqual(self.getFileContent(), content)
        return self.getFileWriter().addCallback(cbGet)