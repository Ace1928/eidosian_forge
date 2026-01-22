import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util
class CommandMixin:
    """
    Tests for all the commands a POP3 server is allowed to receive.
    """
    extraMessage = b'From: guy\nTo: fellow\n\nMore message text for you.\n'

    def setUp(self):
        """
        Make a POP3 server protocol instance hooked up to a simple mailbox and
        a transport that buffers output to a BytesIO.
        """
        p = pop3.POP3()
        p.mbox = self.mailboxType(self.exceptionType)
        p.schedule = list
        self.pop3Server = p
        s = BytesIO()
        p.transport = internet.protocol.FileWrapper(s)
        p.connectionMade()
        s.seek(0)
        s.truncate(0)
        self.pop3Transport = s

    def tearDown(self):
        """
        Disconnect the server protocol so it can clean up anything it might
        need to clean up.
        """
        self.pop3Server.connectionLost(failure.Failure(Exception('Test harness disconnect')))

    def _flush(self):
        """
        Do some of the things that the reactor would take care of, if the
        reactor were actually running.
        """
        self.pop3Server.transport._checkProducer()

    def test_LIST(self):
        """
        Test the two forms of list: with a message index number, which should
        return a short-form response, and without a message index number, which
        should return a long-form response, one line per message.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'LIST 1')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK 1 44\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LIST')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK 1\r\n1 44\r\n.\r\n')

    def test_LISTWithBadArgument(self):
        """
        Test that non-integers and out-of-bound integers produce appropriate
        error responses.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'LIST a')
        self.assertEqual(s.getvalue(), b'-ERR Invalid message-number: a\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LIST 0')
        self.assertEqual(s.getvalue(), b'-ERR Invalid message-number: 0\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LIST 2')
        self.assertEqual(s.getvalue(), b'-ERR Invalid message-number: 2\r\n')
        s.seek(0)
        s.truncate(0)

    def test_UIDL(self):
        """
        Test the two forms of the UIDL command.  These are just like the two
        forms of the LIST command.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'UIDL 1')
        self.assertEqual(s.getvalue(), b'+OK 0\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'UIDL')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK \r\n1 0\r\n.\r\n')

    def test_UIDLWithBadArgument(self):
        """
        Test that UIDL with a non-integer or an out-of-bounds integer produces
        the appropriate error response.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'UIDL a')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'UIDL 0')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'UIDL 2')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)

    def test_STAT(self):
        """
        Test the single form of the STAT command, which returns a short-form
        response of the number of messages in the mailbox and their total size.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'STAT')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK 1 44\r\n')

    def test_RETR(self):
        """
        Test downloading a message.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'RETR 1')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK 44\r\nFrom: moshe\r\nTo: moshe\r\n\r\nHow are you, friend?\r\n.\r\n')
        s.seek(0)
        s.truncate(0)

    def test_RETRWithBadArgument(self):
        """
        Test that trying to download a message with a bad argument, either not
        an integer or an out-of-bounds integer, fails with the appropriate
        error response.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.lineReceived(b'RETR a')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'RETR 0')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'RETR 2')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)

    def test_TOP(self):
        """
        Test downloading the headers and part of the body of a message.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'TOP 1 0')
        self._flush()
        self.assertEqual(s.getvalue(), b'+OK Top of message follows\r\nFrom: moshe\r\nTo: moshe\r\n\r\n.\r\n')

    def test_TOPWithBadArgument(self):
        """
        Test that trying to download a message with a bad argument, either a
        message number which isn't an integer or is an out-of-bounds integer or
        a number of lines which isn't an integer or is a negative integer,
        fails with the appropriate error response.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'TOP 1 a')
        self.assertEqual(s.getvalue(), b'-ERR Bad line count argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'TOP 1 -1')
        self.assertEqual(s.getvalue(), b'-ERR Bad line count argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'TOP a 1')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'TOP 0 1')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'TOP 3 1')
        self.assertEqual(s.getvalue(), b'-ERR Bad message number argument\r\n')
        s.seek(0)
        s.truncate(0)

    def test_LAST(self):
        """
        Test the exceedingly pointless LAST command, which tells you the
        highest message index which you have already downloaded.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 0\r\n')
        s.seek(0)
        s.truncate(0)

    def test_RetrieveUpdatesHighest(self):
        """
        Test that issuing a RETR command updates the LAST response.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'RETR 2')
        self._flush()
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 2\r\n')
        s.seek(0)
        s.truncate(0)

    def test_TopUpdatesHighest(self):
        """
        Test that issuing a TOP command updates the LAST response.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'TOP 2 10')
        self._flush()
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 2\r\n')

    def test_HighestOnlyProgresses(self):
        """
        Test that downloading a message with a smaller index than the current
        LAST response doesn't change the LAST response.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'RETR 2')
        self._flush()
        p.lineReceived(b'TOP 1 10')
        self._flush()
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 2\r\n')

    def test_ResetClearsHighest(self):
        """
        Test that issuing RSET changes the LAST response to 0.
        """
        p = self.pop3Server
        s = self.pop3Transport
        p.mbox.messages.append(self.extraMessage)
        p.lineReceived(b'RETR 2')
        self._flush()
        p.lineReceived(b'RSET')
        s.seek(0)
        s.truncate(0)
        p.lineReceived(b'LAST')
        self.assertEqual(s.getvalue(), b'+OK 0\r\n')