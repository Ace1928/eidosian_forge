import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
class PredefinedCrcTest(unittest.TestCase):
    """Verify the predefined CRCs"""
    test_messages_for_known_answers = [b'', b'T', b'CatMouse989284321']
    known_answers = [['crc-aug-ccitt', (7439, 55021, 22071)], ['x-25', (0, 58585, 2705)], ['crc-32', (0, 3187964512, 139198296)]]

    def test_known_answers(self):
        for crcfun_name, v in self.known_answers:
            crcfun = mkPredefinedCrcFun(crcfun_name)
            self.assertEqual(crcfun(b'', 0), 0, "Wrong answer for CRC '%s', input ''" % crcfun_name)
            for i, msg in enumerate(self.test_messages_for_known_answers):
                self.assertEqual(crcfun(msg), v[i], "Wrong answer for CRC %s, input '%s'" % (crcfun_name, msg))
                self.assertEqual(crcfun(msg[4:], crcfun(msg[:4])), v[i], "Wrong answer for CRC %s, input '%s'" % (crcfun_name, msg))
                self.assertEqual(crcfun(msg[-1:], crcfun(msg[:-1])), v[i], "Wrong answer for CRC %s, input '%s'" % (crcfun_name, msg))

    def test_class_with_known_answers(self):
        for crcfun_name, v in self.known_answers:
            for i, msg in enumerate(self.test_messages_for_known_answers):
                crc1 = PredefinedCrc(crcfun_name)
                crc1.update(msg)
                self.assertEqual(crc1.crcValue, v[i], "Wrong answer for crc1 %s, input '%s'" % (crcfun_name, msg))
                crc2 = crc1.new()
                self.assertEqual(crc1.crcValue, v[i], "Wrong state for crc1 %s, input '%s'" % (crcfun_name, msg))
                self.assertEqual(crc2.crcValue, v[0], "Wrong state for crc2 %s, input '%s'" % (crcfun_name, msg))
                crc2.update(msg)
                self.assertEqual(crc1.crcValue, v[i], "Wrong state for crc1 %s, input '%s'" % (crcfun_name, msg))
                self.assertEqual(crc2.crcValue, v[i], "Wrong state for crc2 %s, input '%s'" % (crcfun_name, msg))

    def test_function_predefined_table(self):
        for table_entry in _predefined_crc_definitions:
            crc_func = mkPredefinedCrcFun(table_entry['name'])
            calc_value = crc_func(b'123456789')
            self.assertEqual(calc_value, table_entry['check'], "Wrong answer for CRC '%s'" % table_entry['name'])

    def test_class_predefined_table(self):
        for table_entry in _predefined_crc_definitions:
            crc1 = PredefinedCrc(table_entry['name'])
            crc1.update(b'123456789')
            self.assertEqual(crc1.crcValue, table_entry['check'], "Wrong answer for CRC '%s'" % table_entry['name'])