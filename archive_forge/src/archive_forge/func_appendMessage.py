import io
import os
import socket
import stat
from hashlib import md5
from typing import IO
from zope.interface import implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, interfaces, reactor
from twisted.mail import mail, pop3, smtp
from twisted.persisted import dirdbm
from twisted.protocols import basic
from twisted.python import failure, log
def appendMessage(self, txt):
    """
        Add a message to the mailbox.

        @type txt: L{bytes} or file-like object
        @param txt: A message to add.

        @rtype: L{Deferred <defer.Deferred>}
        @return: A deferred which fires when the message has been added to
            the mailbox.
        """
    task = self.AppendFactory(self, txt)
    result = task.defer
    task.startUp()
    return result