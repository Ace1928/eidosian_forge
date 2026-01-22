import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def _plaintext(self, username, password):
    """
        Perform a plaintext login.

        @type username: L{bytes}
        @param username: The username with which to log in.

        @type password: L{bytes}
        @param password: The password with which to log in.

        @rtype: L{Deferred <defer.Deferred>} which successfully fires with
            L{bytes} or fails with L{ServerErrorResponse}
        @return: A deferred which fires when the server accepts the username
            and password or fails when the server rejects either.  On a
            successful login, it returns the server's response minus the
            status indicator.
        """
    return self.user(username).addCallback(lambda r: self.password(password))