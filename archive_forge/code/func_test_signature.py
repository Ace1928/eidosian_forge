import unittest
from ..ecc.curve import Curve
from ..util.keyhelper import KeyHelper
def test_signature(self):
    aliceIdentityPublic = bytearray([5, 171, 126, 113, 125, 74, 22, 59, 125, 154, 29, 128, 113, 223, 233, 220, 248, 205, 205, 28, 234, 51, 57, 182, 53, 107, 232, 77, 136, 126, 50, 44, 100])
    aliceEphemeralPublic = bytearray([5, 237, 206, 157, 156, 65, 92, 167, 140, 183, 37, 46, 114, 194, 196, 165, 84, 211, 235, 41, 72, 90, 14, 29, 80, 49, 24, 209, 168, 45, 153, 251, 74])
    aliceSignature = bytearray([93, 232, 140, 169, 168, 155, 74, 17, 93, 167, 145, 9, 198, 124, 156, 116, 100, 163, 228, 24, 2, 116, 241, 203, 140, 99, 194, 152, 78, 40, 109, 251, 237, 232, 45, 235, 157, 205, 159, 174, 11, 251, 184, 33, 86, 155, 61, 144, 1, 189, 129, 48, 205, 17, 212, 134, 206, 240, 71, 189, 96, 184, 110, 136])
    alicePublicKey = Curve.decodePoint(aliceIdentityPublic, 0)
    aliceEphemeral = Curve.decodePoint(aliceEphemeralPublic, 0)
    res = Curve.verifySignature(alicePublicKey, aliceEphemeral.serialize(), bytes(aliceSignature))
    self.assertTrue(res)