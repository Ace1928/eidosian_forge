import unittest
import idna.compat
class IDNACompatTests(unittest.TestCase):

    def testToASCII(self):
        self.assertEqual(idna.compat.ToASCII('テスト.xn--zckzah'), b'xn--zckzah.xn--zckzah')

    def testToUnicode(self):
        self.assertEqual(idna.compat.ToUnicode(b'xn--zckzah.xn--zckzah'), 'テスト.テスト')

    def test_nameprep(self):
        self.assertRaises(NotImplementedError, idna.compat.nameprep, 'a')