import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
class KnownAnswerTests(unittest.TestCase):
    test_messages = [b'T', b'CatMouse989284321']
    known_answers = [[(g8, 0, 0), (254, 157)], [(g8, -1, 1), (79, 155)], [(g8, 0, 1), (254, 98)], [(g16, 0, 0), (6769, 58710)], [(g16, -1, 1), (6950, 62830)], [(g16, 0, 1), (5281, 49805)], [(g24, 0, 0), (12371101, 12891399)], [(g24, -1, 1), (5881102, 698935)], [(g24, 0, 1), (13970191, 1385387)], [(g32, 0, 0), (1804852699, 316449012)], [(g32, 4294967295, 1), (1107002783, 4155768999)], [(g32, 0, 1), (1812370925, 3248754405)], [(g32, 0, 1, 4294967295), (3187964512, 139198296)]]

    def test_known_answers(self):
        for crcfun_params, v in self.known_answers:
            crcfun = mkCrcFun(*crcfun_params)
            self.assertEqual(crcfun(b'', 0), 0, "Wrong answer for CRC parameters %s, input ''" % (crcfun_params,))
            for i, msg in enumerate(self.test_messages):
                self.assertEqual(crcfun(msg), v[i], "Wrong answer for CRC parameters %s, input '%s'" % (crcfun_params, msg))
                self.assertEqual(crcfun(msg[4:], crcfun(msg[:4])), v[i], "Wrong answer for CRC parameters %s, input '%s'" % (crcfun_params, msg))
                self.assertEqual(crcfun(msg[-1:], crcfun(msg[:-1])), v[i], "Wrong answer for CRC parameters %s, input '%s'" % (crcfun_params, msg))