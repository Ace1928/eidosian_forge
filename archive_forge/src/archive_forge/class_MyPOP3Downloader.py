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
class MyPOP3Downloader(pop3.POP3Client):
    """
    A POP3 client which downloads all messages from the server.
    """

    def handle_WELCOME(self, line):
        """
        Authenticate.

        @param line: The welcome response.
        """
        pop3.POP3Client.handle_WELCOME(self, line)
        self.apop(b'hello@baz.com', b'world')

    def handle_APOP(self, line):
        """
        Require an I{OK} response to I{APOP}.

        @param line: The I{APOP} response.
        """
        parts = line.split()
        code = parts[0]
        if code != b'+OK':
            raise AssertionError(f'code is: {code} , parts is: {parts} ')
        self.lines = []
        self.retr(1)

    def handle_RETR_continue(self, line):
        """
        Record one line of message information.

        @param line: A I{RETR} response line.
        """
        self.lines.append(line)

    def handle_RETR_end(self):
        """
        Record the received message information.
        """
        self.message = b'\n'.join(self.lines) + b'\n'
        self.quit()

    def handle_QUIT(self, line):
        """
        Require an I{OK} response to I{QUIT}.

        @param line: The I{QUIT} response.
        """
        if line[:3] != b'+OK':
            raise AssertionError(b'code is ' + line)