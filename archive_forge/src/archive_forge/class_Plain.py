import binascii
import os
import random
import time
from hashlib import md5
from zope.interface import Attribute, Interface, implementer
from twisted.python.compat import networkString
@implementer(ISASLMechanism)
class Plain:
    """
    Implements the PLAIN SASL authentication mechanism.

    The PLAIN SASL authentication mechanism is defined in RFC 2595.
    """
    name = 'PLAIN'

    def __init__(self, authzid, authcid, password):
        """
        @param authzid: The authorization identity.
        @type authzid: L{unicode}

        @param authcid: The authentication identity.
        @type authcid: L{unicode}

        @param password: The plain-text password.
        @type password: L{unicode}
        """
        self.authzid = authzid or ''
        self.authcid = authcid or ''
        self.password = password or ''

    def getInitialResponse(self):
        return self.authzid.encode('utf-8') + b'\x00' + self.authcid.encode('utf-8') + b'\x00' + self.password.encode('utf-8')

    def getResponse(self, challenge):
        pass