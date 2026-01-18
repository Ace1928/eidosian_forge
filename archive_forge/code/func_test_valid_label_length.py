import unittest
import idna
def test_valid_label_length(self):
    self.assertTrue(idna.valid_label_length('a' * 63))
    self.assertFalse(idna.valid_label_length('a' * 64))
    self.assertRaises(idna.IDNAError, idna.encode, 'a' * 64)