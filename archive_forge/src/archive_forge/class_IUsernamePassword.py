import base64
import hmac
import random
import re
import time
from binascii import hexlify
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred import error
from twisted.cred._digest import calcHA1, calcHA2, calcResponse
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.randbytes import secureRandom
from twisted.python.versions import Version
class IUsernamePassword(ICredentials):
    """
    I encapsulate a username and a plaintext password.

    This encapsulates the case where the password received over the network
    has been hashed with the identity function (That is, not at all).  The
    CredentialsChecker may store the password in whatever format it desires,
    it need only transform the stored password in a similar way before
    performing the comparison.

    @type username: L{bytes}
    @ivar username: The username associated with these credentials.

    @type password: L{bytes}
    @ivar password: The password associated with these credentials.
    """
    username: bytes
    password: bytes

    def checkPassword(password: bytes) -> bool:
        """
        Validate these credentials against the correct password.

        @type password: L{bytes}
        @param password: The correct, plaintext password against which to
        check.

        @rtype: C{bool} or L{Deferred}
        @return: C{True} if the credentials represented by this object match the
            given password, C{False} if they do not, or a L{Deferred} which will
            be called back with one of these values.
        """