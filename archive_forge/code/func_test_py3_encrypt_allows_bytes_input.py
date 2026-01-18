import unittest as testm
from sys import version_info
import scrypt
def test_py3_encrypt_allows_bytes_input(self):
    """Test Py3 encrypt allows unicode input."""
    s = scrypt.encrypt(self.byte_text, self.password, 0.1)
    m = scrypt.decrypt(s, self.password)
    self.assertEqual(bytes(m.encode('utf-8')), self.byte_text)