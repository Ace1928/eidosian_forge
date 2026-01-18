import binascii
import os
import random
import time
from hashlib import md5
from zope.interface import Attribute, Interface, implementer
from twisted.python.compat import networkString

        Generate response-value.

        Creates a response to a challenge according to section 2.1.2.1 of
        RFC 2831 using the C{charset}, C{realm} and C{nonce} directives
        from the challenge.
        