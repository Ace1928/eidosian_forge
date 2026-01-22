import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
def _ebMailbox(self, failure):
    """
        Handle an expected authentication failure.

        Send an appropriate error response for a L{LoginDenied} or
        L{LoginFailed} authentication failure.

        @type failure: L{Failure}
        @param failure: The authentication error.
        """
    failure = failure.trap(cred.error.LoginDenied, cred.error.LoginFailed)
    if issubclass(failure, cred.error.LoginDenied):
        self.failResponse('Access denied: ' + str(failure))
    elif issubclass(failure, cred.error.LoginFailed):
        self.failResponse(b'Authentication failed')
    if getattr(self.factory, 'noisy', True):
        log.msg('Denied login attempt from ' + str(self.transport.getPeer()))