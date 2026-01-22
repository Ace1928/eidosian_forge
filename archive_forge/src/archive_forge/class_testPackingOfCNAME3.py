import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfCNAME3(PackerTestCase):
    """addCNAME(self, name, klass, ttl, cname)"""

    def doPack(self, p):
        p.addCNAME('www.fred.com', DNS.Class.IN, 86400, 'webhost.loa.com')

    def doUnpack(self, u):
        return u.getCNAMEdata()
    unpackerExpectedResult = (('www.fred.com', DNS.Type.CNAME, DNS.Class.IN, 86400), 'webhost.loa.com')
    packerExpectedResult = b'\x03www\x04fred\x03com\x00\x00\x05\x00\x01\x00\x01Q' + b'\x80\x00\x0e\x07webhost\x03loa\xc0\t'