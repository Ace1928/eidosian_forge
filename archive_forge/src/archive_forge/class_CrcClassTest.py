import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
class CrcClassTest(unittest.TestCase):
    """Verify the Crc class"""
    msg = b'CatMouse989284321'

    def test_simple_crc32_class(self):
        """Verify the CRC class when not using xorOut"""
        crc = Crc(g32)
        str_rep = 'poly = 0x104C11DB7\nreverse = True\ninitCrc  = 0xFFFFFFFF\nxorOut   = 0x00000000\ncrcValue = 0xFFFFFFFF'
        self.assertEqual(str(crc), str_rep)
        self.assertEqual(crc.digest(), b'\xff\xff\xff\xff')
        self.assertEqual(crc.hexdigest(), 'FFFFFFFF')
        crc.update(self.msg)
        self.assertEqual(crc.crcValue, 4155768999)
        self.assertEqual(crc.digest(), b'\xf7\xb4\x00\xa7')
        self.assertEqual(crc.hexdigest(), 'F7B400A7')
        x = crc.copy()
        self.assertTrue(x is not crc)
        str_rep = 'poly = 0x104C11DB7\nreverse = True\ninitCrc  = 0xFFFFFFFF\nxorOut   = 0x00000000\ncrcValue = 0xF7B400A7'
        self.assertEqual(str(crc), str_rep)
        self.assertEqual(str(x), str_rep)

    def test_full_crc32_class(self):
        """Verify the CRC class when using xorOut"""
        crc = Crc(g32, initCrc=0, xorOut=~0)
        str_rep = 'poly = 0x104C11DB7\nreverse = True\ninitCrc  = 0x00000000\nxorOut   = 0xFFFFFFFF\ncrcValue = 0x00000000'
        self.assertEqual(str(crc), str_rep)
        self.assertEqual(crc.digest(), b'\x00\x00\x00\x00')
        self.assertEqual(crc.hexdigest(), '00000000')
        crc.update(self.msg)
        self.assertEqual(crc.crcValue, 139198296)
        self.assertEqual(crc.digest(), b'\x08K\xffX')
        self.assertEqual(crc.hexdigest(), '084BFF58')
        x = crc.copy()
        self.assertTrue(x is not crc)
        str_rep = 'poly = 0x104C11DB7\nreverse = True\ninitCrc  = 0x00000000\nxorOut   = 0xFFFFFFFF\ncrcValue = 0x084BFF58'
        self.assertEqual(str(crc), str_rep)
        self.assertEqual(str(x), str_rep)
        y = crc.new()
        self.assertTrue(y is not crc)
        self.assertTrue(y is not x)
        str_rep = 'poly = 0x104C11DB7\nreverse = True\ninitCrc  = 0x00000000\nxorOut   = 0xFFFFFFFF\ncrcValue = 0x00000000'
        self.assertEqual(str(y), str_rep)