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
class DummyMailbox(pop3.Mailbox):
    """
    An in-memory L{pop3.IMailbox} implementation.

    @ivar messages: A sequence of L{bytes} defining the messages in this
        mailbox.

    @ivar exceptionType: The type of exception to raise when an out-of-bounds
        index is addressed.
    """
    messages = [b'From: moshe\nTo: moshe\n\nHow are you, friend?\n']

    def __init__(self, exceptionType):
        self.messages = DummyMailbox.messages[:]
        self.exceptionType = exceptionType

    def listMessages(self, i=None):
        """
        Get some message information.

        @param i: See L{pop3.IMailbox.listMessages}.
        @return: See L{pop3.IMailbox.listMessages}.
        """
        if i is None:
            return [len(m) for m in self.messages]
        if i >= len(self.messages):
            raise self.exceptionType()
        return len(self.messages[i])

    def getMessage(self, i):
        """
        Get the message content.

        @param i: See L{pop3.IMailbox.getMessage}.
        @return: See L{pop3.IMailbox.getMessage}.
        """
        return BytesIO(self.messages[i])

    def getUidl(self, i):
        """
        Construct a UID which is simply the string representation of the given
        index.

        @param i: See L{pop3.IMailbox.getUidl}.
        @return: See L{pop3.IMailbox.getUidl}.
        """
        if i >= len(self.messages):
            raise self.exceptionType()
        return b'%d' % (i,)

    def deleteMessage(self, i):
        """
        Wipe the message at the given index.

        @param i: See L{pop3.IMailbox.deleteMessage}.
        """
        self.messages[i] = b''