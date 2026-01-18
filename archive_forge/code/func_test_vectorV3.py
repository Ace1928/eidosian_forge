import unittest
from ...kdf.hkdf import HKDF
def test_vectorV3(self):
    ikm = bytearray([11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11])
    salt = bytearray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    info = bytearray([240, 241, 242, 243, 244, 245, 246, 247, 248, 249])
    okm = bytearray([60, 178, 95, 37, 250, 172, 213, 122, 144, 67, 79, 100, 208, 54, 47, 42, 45, 45, 10, 144, 207, 26, 90, 76, 93, 176, 45, 86, 236, 196, 197, 191, 52, 0, 114, 8, 213, 184, 135, 24, 88, 101])
    actualOutput = HKDF.createFor(3).deriveSecrets(ikm, info, 42, salt=salt)
    self.assertEqual(okm, actualOutput)