import unittest as testm
from sys import version_info
import scrypt
def test_py3_encrypt_allows_bytes_password(self):
    """Test Py3 encrypt allows unicode password."""
    s = scrypt.encrypt(self.input, self.byte_text, 0.1)
    m = scrypt.decrypt(s, self.byte_text)
    self.assertEqual(m, self.input)