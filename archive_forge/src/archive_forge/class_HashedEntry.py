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
@implementer(IKnownHostEntry)
class HashedEntry(_BaseEntry, FancyEqMixin):
    """
    A L{HashedEntry} is a representation of an entry in a known_hosts file
    where the hostname has been hashed and salted.

    @ivar _hostSalt: the salt to combine with a hostname for hashing.

    @ivar _hostHash: the hashed representation of the hostname.

    @cvar MAGIC: the 'hash magic' string used to identify a hashed line in a
    known_hosts file as opposed to a plaintext one.
    """
    MAGIC = b'|1|'
    compareAttributes = ('_hostSalt', '_hostHash', 'keyType', 'publicKey', 'comment')

    def __init__(self, hostSalt, hostHash, keyType, publicKey, comment):
        self._hostSalt = hostSalt
        self._hostHash = hostHash
        super().__init__(keyType, publicKey, comment)

    @classmethod
    def fromString(cls, string):
        """
        Load a hashed entry from a string representing a line in a known_hosts
        file.

        @param string: A complete single line from a I{known_hosts} file,
            formatted as defined by OpenSSH.
        @type string: L{bytes}

        @raise DecodeError: if the key, the hostname, or the is not valid
            encoded as valid base64

        @raise InvalidEntry: if the entry does not have the right number of
            elements and is therefore invalid, or the host/hash portion contains
            more items than just the host and hash.

        @raise BadKeyError: if the key, once decoded from base64, is not
            actually an SSH key.

        @return: The newly created L{HashedEntry} instance, initialized with the
            information from C{string}.
        """
        stuff, keyType, key, comment = _extractCommon(string)
        saltAndHash = stuff[len(cls.MAGIC):].split(b'|')
        if len(saltAndHash) != 2:
            raise InvalidEntry()
        hostSalt, hostHash = saltAndHash
        self = cls(a2b_base64(hostSalt), a2b_base64(hostHash), keyType, key, comment)
        return self

    def matchesHost(self, hostname):
        """
        Implement L{IKnownHostEntry.matchesHost} to compare the hash of the
        input to the stored hash.

        @param hostname: A hostname or IP address literal to check against this
            entry.
        @type hostname: L{bytes}

        @return: C{True} if this entry is for the given hostname or IP address,
            C{False} otherwise.
        @rtype: L{bool}
        """
        return hmac.compare_digest(_hmacedString(self._hostSalt, hostname), self._hostHash)

    def toString(self):
        """
        Implement L{IKnownHostEntry.toString} by base64-encoding the salt, host
        hash, and key.

        @return: The string representation of this entry, with the hostname part
            hashed.
        @rtype: L{bytes}
        """
        fields = [self.MAGIC + b'|'.join([_b64encode(self._hostSalt), _b64encode(self._hostHash)]), self.keyType, _b64encode(self.publicKey.blob())]
        if self.comment is not None:
            fields.append(self.comment)
        return b' '.join(fields)