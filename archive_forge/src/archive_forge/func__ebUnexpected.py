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
def _ebUnexpected(self, failure):
    """
        Handle an unexpected authentication failure.

        Send an error response for an unexpected authentication failure.

        @type failure: L{Failure}
        @param failure: The authentication error.
        """
    self.failResponse('Server error: ' + failure.getErrorMessage())
    log.err(failure)