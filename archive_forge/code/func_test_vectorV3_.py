import unittest
from ...kdf.hkdf import HKDF
def test_vectorV3_(self):
    ikm = bytearray([11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11])
    salt = bytearray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    info = bytearray([240, 241, 242, 243, 244, 245, 246, 247, 248, 249])
    okm = bytearray([110, 194, 85, 109, 93, 123, 29, 129, 222, 228, 34, 42, 215, 72, 54, 149, 221, 201, 143, 79, 95, 171, 192, 224, 32, 93, 194, 239, 135, 82, 212, 30, 4, 226, 226, 17, 1, 198, 143, 240, 147, 148, 184, 173, 11, 220, 185, 96, 156, 212, 238, 130, 172, 19, 25, 155, 74, 169, 253, 168, 153, 218, 235, 236])
    actualOutput = HKDF.createFor(2).deriveSecrets(ikm, info, 64, salt=salt)
    self.assertEqual(okm, actualOutput)