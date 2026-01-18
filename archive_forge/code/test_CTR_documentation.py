import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util import Counter
Class exercising the CTR test vectors found in Section F.5
    of NIST SP 800-38A