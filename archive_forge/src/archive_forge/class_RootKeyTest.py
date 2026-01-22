import unittest
from ...ecc.curve import Curve
from ...ecc.eckeypair import ECKeyPair
from ...ratchet.rootkey import RootKey
from ...kdf.hkdf import HKDF
class RootKeyTest(unittest.TestCase):

    def test_rootKeyDerivationV2(self):
        rootKeySeed = bytearray([123, 166, 222, 188, 43, 193, 187, 249, 26, 187, 193, 54, 116, 4, 23, 108, 166, 35, 9, 91, 126, 198, 107, 69, 246, 2, 217, 53, 56, 148, 45, 204])
        alicePublic = bytearray([5, 238, 79, 166, 205, 192, 48, 223, 73, 236, 208, 186, 108, 252, 255, 178, 51, 211, 101, 162, 127, 173, 190, 255, 119, 233, 99, 252, 177, 98, 34, 225, 58])
        alicePrivate = bytearray([33, 104, 34, 236, 103, 235, 56, 4, 158, 186, 231, 185, 57, 186, 234, 235, 177, 81, 187, 179, 45, 184, 15, 211, 137, 36, 90, 195, 122, 148, 142, 80])
        bobPublic = bytearray([5, 171, 184, 235, 41, 204, 128, 180, 113, 9, 162, 38, 90, 190, 151, 152, 72, 84, 6, 227, 45, 162, 104, 147, 74, 149, 85, 232, 71, 87, 112, 138, 48])
        nextRoot = bytearray([177, 20, 245, 222, 40, 1, 25, 133, 230, 235, 162, 93, 80, 231, 236, 65, 169, 176, 47, 86, 147, 197, 199, 136, 166, 58, 6, 210, 18, 162, 247, 49])
        nextChain = bytearray([157, 125, 36, 105, 188, 154, 229, 62, 233, 128, 90, 163, 38, 77, 36, 153, 163, 172, 232, 15, 76, 202, 226, 218, 19, 67, 12, 92, 85, 181, 202, 95])
        alicePublicKey = Curve.decodePoint(alicePublic, 0)
        alicePrivateKey = Curve.decodePrivatePoint(alicePrivate)
        aliceKeyPair = ECKeyPair(alicePublicKey, alicePrivateKey)
        bobPublicKey = Curve.decodePoint(bobPublic, 0)
        rootKey = RootKey(HKDF.createFor(2), rootKeySeed)
        rootKeyChainKeyPair = rootKey.createChain(bobPublicKey, aliceKeyPair)
        nextRootKey = rootKeyChainKeyPair[0]
        nextChainKey = rootKeyChainKeyPair[1]
        self.assertEqual(rootKey.getKeyBytes(), rootKeySeed)
        self.assertEqual(nextRootKey.getKeyBytes(), nextRoot)
        self.assertEqual(nextChainKey.getKey(), nextChain)