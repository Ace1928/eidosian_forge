import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfNS(PackerTestCase):
    """addNS(self, name, klass, ttl, nsdname)"""

    def doPack(self, p):
        p.addNS('ekit-inc.com', DNS.Class.IN, 86400, 'ns1.ekorp.com')

    def doUnpack(self, u):
        return u.getNSdata()
    unpackerExpectedResult = (('ekit-inc.com', 2, 1, 86400), 'ns1.ekorp.com')
    packerExpectedResult = b'\x08ekit-inc\x03com\x00\x00\x02\x00\x01\x00\x01Q\x80\x00\x0c\x03ns1\x05ekorp\xc0\t'