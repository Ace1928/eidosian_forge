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
class MockFactory(factory.SSHFactory):
    """
    A mocked-up factory based on twisted.conch.ssh.factory.SSHFactory.
    """
    services = {b'ssh-userauth': MockService}

    def getPublicKeys(self):
        """
        Return the public keys that authenticate this server.
        """
        return {b'ssh-rsa': keys.Key.fromString(keydata.publicRSA_openssh), b'ssh-dsa': keys.Key.fromString(keydata.publicDSA_openssh)}

    def getPrivateKeys(self):
        """
        Return the private keys that authenticate this server.
        """
        return {b'ssh-rsa': keys.Key.fromString(keydata.privateRSA_openssh), b'ssh-dsa': keys.Key.fromString(keydata.privateDSA_openssh)}

    def getPrimes(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Diffie-Hellman primes that can be used for key exchange algorithms
        that use group exchange to establish a prime / generator group.

        @return: The primes and generators.
        @rtype: L{dict} mapping the key size to a C{list} of
            C{(generator, prime)} tuple.
        """
        group14 = _kex.getDHGeneratorAndPrime(b'diffie-hellman-group14-sha1')
        return {2048: [group14], 4096: [(5, 7)]}