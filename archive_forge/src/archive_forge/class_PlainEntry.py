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
class PlainEntry(_BaseEntry):
    """
    A L{PlainEntry} is a representation of a plain-text entry in a known_hosts
    file.

    @ivar _hostnames: the list of all host-names associated with this entry.
    @type _hostnames: L{list} of L{bytes}
    """

    def __init__(self, hostnames, keyType, publicKey, comment):
        self._hostnames = hostnames
        super().__init__(keyType, publicKey, comment)

    @classmethod
    def fromString(cls, string):
        """
        Parse a plain-text entry in a known_hosts file, and return a
        corresponding L{PlainEntry}.

        @param string: a space-separated string formatted like "hostname
        key-type base64-key-data comment".

        @type string: L{bytes}

        @raise DecodeError: if the key is not valid encoded as valid base64.

        @raise InvalidEntry: if the entry does not have the right number of
        elements and is therefore invalid.

        @raise BadKeyError: if the key, once decoded from base64, is not
        actually an SSH key.

        @return: an IKnownHostEntry representing the hostname and key in the
        input line.

        @rtype: L{PlainEntry}
        """
        hostnames, keyType, key, comment = _extractCommon(string)
        self = cls(hostnames.split(b','), keyType, key, comment)
        return self

    def matchesHost(self, hostname):
        """
        Check to see if this entry matches a given hostname.

        @param hostname: A hostname or IP address literal to check against this
            entry.
        @type hostname: L{bytes}

        @return: C{True} if this entry is for the given hostname or IP address,
            C{False} otherwise.
        @rtype: L{bool}
        """
        if isinstance(hostname, str):
            hostname = hostname.encode('utf-8')
        return hostname in self._hostnames

    def toString(self):
        """
        Implement L{IKnownHostEntry.toString} by recording the comma-separated
        hostnames, key type, and base-64 encoded key.

        @return: The string representation of this entry, with unhashed hostname
            information.
        @rtype: L{bytes}
        """
        fields = [b','.join(self._hostnames), self.keyType, _b64encode(self.publicKey.blob())]
        if self.comment is not None:
            fields.append(self.comment)
        return b' '.join(fields)