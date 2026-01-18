import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import *
from Cryptodome.IO import PKCS8
from Cryptodome.Util.asn1 import DerNull
def txt2bin(inputs):
    s = b('').join([b(x) for x in inputs if not x in '\n\r\t '])
    return unhexlify(s)