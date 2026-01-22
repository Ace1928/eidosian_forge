import binascii
import os
import random
import time
from hashlib import md5
from zope.interface import Attribute, Interface, implementer
from twisted.python.compat import networkString
def _unparse(self, directives):
    """
        Create message string from directives.

        @param directives: dictionary of directives (names to their values).
                           For certain directives, extra quotes are added, as
                           needed.
        @type directives: C{dict} of C{str} to C{str}
        @return: message string.
        @rtype: C{str}.
        """
    directive_list = []
    for name, value in directives.items():
        if name in (b'username', b'realm', b'cnonce', b'nonce', b'digest-uri', b'authzid', b'cipher'):
            directive = name + b'=' + value
        else:
            directive = name + b'=' + value
        directive_list.append(directive)
    return b','.join(directive_list)