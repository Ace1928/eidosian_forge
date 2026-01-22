import hmac
import sys
from binascii import Error as DecodeError, a2b_base64, b2a_base64
from contextlib import closing
from hashlib import sha1
from zope.interface import implementer
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.conch.ssh.keys import BadKeyError, FingerprintFormats, Key
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString
from twisted.python.randbytes import secureRandom
from twisted.python.util import FancyEqMixin
def _extractCommon(string):
    """
    Extract common elements of base64 keys from an entry in a hosts file.

    @param string: A known hosts file entry (a single line).
    @type string: L{bytes}

    @return: a 4-tuple of hostname data (L{bytes}), ssh key type (L{bytes}), key
        (L{Key}), and comment (L{bytes} or L{None}).  The hostname data is
        simply the beginning of the line up to the first occurrence of
        whitespace.
    @rtype: L{tuple}
    """
    elements = string.split(None, 2)
    if len(elements) != 3:
        raise InvalidEntry()
    hostnames, keyType, keyAndComment = elements
    splitkey = keyAndComment.split(None, 1)
    if len(splitkey) == 2:
        keyString, comment = splitkey
        comment = comment.rstrip(b'\n')
    else:
        keyString = splitkey[0]
        comment = None
    key = Key.fromString(a2b_base64(keyString))
    return (hostnames, keyType, key, comment)