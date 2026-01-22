import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testPackingOfHINFO(PackerTestCase):
    """addHINFO(self, name, klass, ttl, cpu, os)"""

    def doPack(self, p):
        p.addHINFO('www.sub.domain.com', DNS.Class.IN, 3600, 'i686', 'linux')

    def doUnpack(self, u):
        return u.getHINFOdata()
    unpackerExpectedResult = (('www.sub.domain.com', 13, 1, 3600), ('i686', 'linux'))
    packerExpectedResult = b'\x03www\x03sub\x06domain\x03com\x00\x00\r\x00\x01' + b'\x00\x00\x0e\x10\x00\x0b\x04i686\x05linux'