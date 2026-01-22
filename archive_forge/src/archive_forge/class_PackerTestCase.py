import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
class PackerTestCase(unittest.TestCase):
    """ base class for tests of Packing code. Laziness on my part, I know. """

    def setUp(self):
        self.RRpacker = DNS.Lib.RRpacker
        self.RRunpacker = DNS.Lib.RRunpacker

    def testPacker(self):
        p = self.RRpacker()
        check = self.doPack(p)
        if p is not None and check is not TestCompleted:
            return self.checkPackResult(p)

    def checkPackResult(self, buf):
        if not hasattr(self, 'packerExpectedResult'):
            if self.__class__.__name__ != 'PackerTestCase':
                print('P***', self, repr(buf.getbuf()))
        else:
            return self.assertEqual(buf.getbuf(), self.packerExpectedResult)

    def checkUnpackResult(self, rrbits, specbits):
        if not hasattr(self, 'unpackerExpectedResult'):
            if self.__class__.__name__ != 'PackerTestCase':
                print('U***', self, repr((rrbits, specbits)))
        else:
            return self.assertEqual((rrbits, specbits), self.unpackerExpectedResult)

    def testUnpacker(self):
        if self.doUnpack is not None:
            if hasattr(self.__class__, 'doUnpack') and hasattr(self, 'packerExpectedResult'):
                u = self.RRunpacker(self.packerExpectedResult)
                rrbits = u.getRRheader()[:4]
                specbits = self.doUnpack(u)
                try:
                    u.endRR()
                except DNS.Lib.UnpackError:
                    self.assertEqual(0, 'Not at end of RR!')
                return self.checkUnpackResult(rrbits, specbits)
            else:
                me = self.__class__.__name__
                if me != 'PackerTestCase':
                    self.assertEquals(self.__class__.__name__, 'Unpack NotImplemented')

    def doPack(self, p):
        """ stub. don't test the base class """
        return None

    def doUnpack(self, p):
        """ stub. don't test the base class """
        return None