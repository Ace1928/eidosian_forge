import binascii
import os
import random
import time
from hashlib import md5
from zope.interface import Attribute, Interface, implementer
from twisted.python.compat import networkString
def _calculateResponse(self, cnonce, nc, nonce, username, password, realm, uri):
    """
        Calculates response with given encoded parameters.

        @return: The I{response} field of a response to a Digest-MD5 challenge
            of the given parameters.
        @rtype: L{bytes}
        """

    def H(s):
        return md5(s).digest()

    def HEX(n):
        return binascii.b2a_hex(n)

    def KD(k, s):
        return H(k + b':' + s)
    a1 = H(username + b':' + realm + b':' + password) + b':' + nonce + b':' + cnonce
    a2 = b'AUTHENTICATE:' + uri
    response = HEX(KD(HEX(H(a1)), nonce + b':' + nc + b':' + cnonce + b':' + b'auth' + b':' + HEX(H(a2))))
    return response