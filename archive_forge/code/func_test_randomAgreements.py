import unittest
from ..ecc.curve import Curve
from ..util.keyhelper import KeyHelper
def test_randomAgreements(self):
    for i in range(0, 50):
        alice = Curve.generateKeyPair()
        bob = Curve.generateKeyPair()
        sharedAlice = Curve.calculateAgreement(bob.getPublicKey(), alice.getPrivateKey())
        sharedBob = Curve.calculateAgreement(alice.getPublicKey(), bob.getPrivateKey())
        self.assertEqual(sharedAlice, sharedBob)