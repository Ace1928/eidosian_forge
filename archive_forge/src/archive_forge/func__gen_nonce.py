import binascii
import os
import random
import time
from hashlib import md5
from zope.interface import Attribute, Interface, implementer
from twisted.python.compat import networkString
def _gen_nonce(self):
    nonceString = '%f:%f:%d' % (random.random(), time.time(), os.getpid())
    nonceBytes = networkString(nonceString)
    return md5(nonceBytes).hexdigest().encode('ascii')