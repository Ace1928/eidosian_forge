import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfTXT2(PackerTestCase):
    """addTXT(self, name, klass, ttl, list)"""

    def doPack(self, p):
        f = lambda p=p: p.addTXT('ekit-inc.com', DNS.Class.IN, 3600, 'the quick brown fox jumped over the lazy brown dog\n' * 20)
        self.assertRaises(ValueError, f)
        return TestCompleted
    doUnpack = None