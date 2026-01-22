import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class IPaddrPacking(unittest.TestCase):
    knownValues = (('127.0.0.1', 2130706433), ('10.99.23.13', 174266125), ('192.35.59.45', 3223534381), ('255.255.255.255', 4294967295))

    def testIPaddrPacking(self):
        """ addr2bin should give known output for known input """
        for i, s in self.knownValues:
            result = DNS.Lib.addr2bin(i)
            self.assertEqual(s, result)

    def testIPaddrUnpacking(self):
        """ bin2addr should give known output for known input """
        for i, s in self.knownValues:
            result = DNS.Lib.bin2addr(s)
            self.assertEqual(i, result)