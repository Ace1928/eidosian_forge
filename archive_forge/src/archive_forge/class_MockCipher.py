import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class MockCipher:
    """
    A mocked-up version of twisted.conch.ssh.transport.SSHCiphers.
    """
    outCipType = b'test'
    encBlockSize = 6
    inCipType = b'test'
    decBlockSize = 6
    inMACType = b'test'
    outMACType = b'test'
    verifyDigestSize = 1
    usedEncrypt = False
    usedDecrypt = False
    outMAC = (None, b'', b'', 1)
    inMAC = (None, b'', b'', 1)
    keys = ()

    def encrypt(self, x):
        """
        Called to encrypt the packet.  Simply record that encryption was used
        and return the data unchanged.
        """
        self.usedEncrypt = True
        if len(x) % self.encBlockSize != 0:
            raise RuntimeError('length %i modulo blocksize %i is not 0: %i' % (len(x), self.encBlockSize, len(x) % self.encBlockSize))
        return x

    def decrypt(self, x):
        """
        Called to decrypt the packet.  Simply record that decryption was used
        and return the data unchanged.
        """
        self.usedDecrypt = True
        if len(x) % self.encBlockSize != 0:
            raise RuntimeError('length %i modulo blocksize %i is not 0: %i' % (len(x), self.decBlockSize, len(x) % self.decBlockSize))
        return x

    def makeMAC(self, outgoingPacketSequence, payload):
        """
        Make a Message Authentication Code by sending the character value of
        the outgoing packet.
        """
        return bytes((outgoingPacketSequence,))

    def verify(self, incomingPacketSequence, packet, macData):
        """
        Verify the Message Authentication Code by checking that the packet
        sequence number is the same.
        """
        return bytes((incomingPacketSequence,)) == macData

    def setKeys(self, ivOut, keyOut, ivIn, keyIn, macIn, macOut):
        """
        Record the keys.
        """
        self.keys = (ivOut, keyOut, ivIn, keyIn, macIn, macOut)