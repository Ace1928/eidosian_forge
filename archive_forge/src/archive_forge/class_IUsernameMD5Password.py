import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
class IUsernameMD5Password(ICredentials):
    """
    I encapsulate a username and a hashed password.

    This credential is used for username/password over PB. CredentialCheckers
    which check this kind of credential must store the passwords in plaintext
    form or as a MD5 digest.

    @type username: C{str} or C{Deferred}
    @ivar username: The username associated with these credentials.
    """

    def checkPassword(password):
        """
        Validate these credentials against the correct password.

        @type password: C{str}
        @param password: The correct, plaintext password against which to
            check.

        @rtype: C{bool} or L{Deferred}
        @return: C{True} if the credentials represented by this object match the
            given password, C{False} if they do not, or a L{Deferred} which will
            be called back with one of these values.
        """

    def checkMD5Password(password):
        """
        Validate these credentials against the correct MD5 digest of the
        password.

        @type password: C{str}
        @param password: The correct MD5 digest of a password against which to
            check.

        @rtype: C{bool} or L{Deferred}
        @return: C{True} if the credentials represented by this object match the
            given digest, C{False} if they do not, or a L{Deferred} which will
            be called back with one of these values.
        """