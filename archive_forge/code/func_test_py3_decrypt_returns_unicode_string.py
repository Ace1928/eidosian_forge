import unittest as testm
from sys import version_info
import scrypt
def test_py3_decrypt_returns_unicode_string(self):
    """Test Py3 decrypt returns Unicode UTF-8 string."""
    s = scrypt.encrypt(self.input, self.password, 0.1)
    m = scrypt.decrypt(s, self.password)
    self.assertTrue(isinstance(m, str))