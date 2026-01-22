import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfCNAME(PackerTestCase):
    """addCNAME(self, name, klass, ttl, cname)"""

    def doPack(self, p):
        p.addCNAME('www.sub.domain', DNS.Class.IN, 3600, 'realhost.sub.domain')

    def doUnpack(self, u):
        return u.getCNAMEdata()
    unpackerExpectedResult = (('www.sub.domain', DNS.Type.CNAME, DNS.Class.IN, 3600), 'realhost.sub.domain')
    packerExpectedResult = b'\x03www\x03sub\x06domain\x00\x00\x05\x00\x01\x00' + b'\x00\x0e\x10\x00\x0b\x08realhost\xc0\x04'