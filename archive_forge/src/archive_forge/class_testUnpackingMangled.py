import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class testUnpackingMangled(unittest.TestCase):
    """addA(self, name, klass, ttl, address)"""
    packerCorrect = b'\x05www02\x04ekit\x03com\x00\x00\x01\x00\x01\x00\x01Q\x80\x00\x04\xc0\xa8\n\x02'

    def testWithoutRR(self):
        u = DNS.Lib.RRunpacker(self.packerCorrect)
        u.getAdata()

    def testWithTwoRRs(self):
        u = DNS.Lib.RRunpacker(self.packerCorrect)
        u.getRRheader()
        self.assertRaises(DNS.Lib.UnpackError, u.getRRheader)

    def testWithNoGetData(self):
        u = DNS.Lib.RRunpacker(self.packerCorrect)
        u.getRRheader()
        self.assertRaises(DNS.Lib.UnpackError, u.endRR)