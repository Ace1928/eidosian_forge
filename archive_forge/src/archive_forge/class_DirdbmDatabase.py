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
@implementer(checkers.ICredentialsChecker)
class DirdbmDatabase:
    """
    A credentials checker which authenticates users out of a
    L{DirDBM <dirdbm.DirDBM>} database.

    @type dirdbm: L{DirDBM <dirdbm.DirDBM>}
    @ivar dirdbm: An authentication database.
    """
    credentialInterfaces = (credentials.IUsernamePassword, credentials.IUsernameHashedPassword)

    def __init__(self, dbm):
        """
        @type dbm: L{DirDBM <dirdbm.DirDBM>}
        @param dbm: An authentication database.
        """
        self.dirdbm = dbm

    def requestAvatarId(self, c):
        """
        Authenticate a user and, if successful, return their username.

        @type c: L{IUsernamePassword <credentials.IUsernamePassword>} or
            L{IUsernameHashedPassword <credentials.IUsernameHashedPassword>}
            provider.
        @param c: Credentials.

        @rtype: L{bytes}
        @return: A string which identifies an user.

        @raise UnauthorizedLogin: When the credentials check fails.
        """
        if c.username in self.dirdbm:
            if c.checkPassword(self.dirdbm[c.username]):
                return c.username
        raise UnauthorizedLogin()