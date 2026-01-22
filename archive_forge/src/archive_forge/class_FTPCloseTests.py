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
class FTPCloseTests(TestCase):
    """
    Tests that the server invokes IWriteFile.close
    """

    def test_write(self):
        """
        Confirm that FTP uploads (i.e. ftp_STOR) correctly call and wait
        upon the IWriteFile object's close() method
        """
        f = ftp.FTP()
        f.workingDirectory = ['root']
        f.shell = CloseTestShell()
        f.shell.writer = CloseTestWriter()
        f.shell.writer.d = defer.Deferred()
        f.factory = ftp.FTPFactory()
        f.factory.timeOut = None
        f.makeConnection(BytesIO())
        di = ftp.DTP()
        di.factory = ftp.DTPFactory(f)
        f.dtpInstance = di
        di.makeConnection(None)
        stor_done = []
        d = f.ftp_STOR('path')
        d.addCallback(stor_done.append)
        self.assertFalse(f.shell.writer.closeStarted, 'close() called early')
        di.dataReceived(b'some data here')
        self.assertFalse(f.shell.writer.closeStarted, 'close() called early')
        di.connectionLost('reason is ignored')
        self.assertTrue(f.shell.writer.closeStarted, 'close() not called')
        self.assertFalse(stor_done)
        f.shell.writer.d.callback('allow close() to finish')
        self.assertTrue(stor_done)
        return d