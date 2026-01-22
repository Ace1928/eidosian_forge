import re
import unittest
from binascii import hexlify
from Cryptodome.Util.py3compat import bord
from Cryptodome.Hash import SHA256
from Cryptodome.PublicKey import ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Protocol.DH import key_agreement
class ECDH_Tests(unittest.TestCase):
    static_priv = ECC.import_key('-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg9VHFVKh2a1aVFifH\n+BiyNaRa2kttEg3165Ye/dJxJ7KhRANCAARImIEXro5ZOcyWU2mq/+d79FEZXtTA\nbKkz1aICQXihQdCMzRNbeNtC9LFLzhu1slRKJ2xsDAlw9r6w6vwtkRzr\n-----END PRIVATE KEY-----')
    static_pub = ECC.import_key('-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgHhmv8zmZ+Nw8fsZd\ns8tlZflyfw2NE1CRS9DWr3Y3O46hRANCAAS3hZVUCbk+uk3w4S/YOraEVGG+WYpk\nNO/vrwzufUUks2GV2OnBQESe0EBk4Jq8gn4ij8Lvs3rZX2yT+XfeATYd\n-----END PRIVATE KEY-----').public_key()
    eph_priv = ECC.import_key('-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgGPdJmFFFKzLPspIr\nE1T2cEjeIf4ajS9CpneP0e2b3AyhRANCAAQBexAA5BYDcXHs2KOksTYUsst4HhPt\nkp0zkgI2virc3OGJFNGPaCCPfFCQJHwLRaEpiq3SoQlgoBwSc8ZPsl3y\n-----END PRIVATE KEY-----')
    eph_pub = ECC.import_key('-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQghaVZXElSEGEojFKF\nOU0JCpxWUWHvWQUR81gwWrOp76ShRANCAATi1Ib2K+YR3AckD8wxypWef7pw5PRw\ntBaB3RDPyE7IjHZC6yu1DbcXoCdtaw+F5DM+4zpl59n5ZaIy/Yl1BdIy\n-----END PRIVATE KEY-----')

    def test_1(self):
        kdf = lambda x: SHA256.new(x).digest()
        z = key_agreement(kdf=kdf, static_pub=self.static_pub, static_priv=self.static_priv)
        self.assertEqual(hexlify(z), b'3960a1101d1193cbaffef4cc7202ebff783c22c6d2e0d5d530ffc66dc197ea9c')

    def test_2(self):
        kdf = lambda x: SHA256.new(x).digest()
        z = key_agreement(kdf=kdf, static_pub=self.static_pub, static_priv=self.static_priv, eph_pub=self.eph_pub, eph_priv=self.eph_priv)
        self.assertEqual(hexlify(z), b'7447b733d40c8fab2c633b3dc61e4a8c742f3a6af7e16fb0cc486f5bdb5d6ba2')

    def test_3(self):
        kdf = lambda x: SHA256.new(x).digest()
        z = key_agreement(kdf=kdf, static_pub=self.static_pub, static_priv=self.static_priv, eph_priv=self.eph_priv)
        self.assertEqual(hexlify(z), b'9e977ae45f33bf67f285d064d83e6632bcafe3a7d33fe571233bab4794ace759')

    def test_4(self):
        kdf = lambda x: SHA256.new(x).digest()
        z = key_agreement(kdf=kdf, static_pub=self.static_pub, static_priv=self.static_priv, eph_pub=self.eph_pub)
        self.assertEqual(hexlify(z), b'c9532df6aa7e9dbe5fe85da31ee25ff19c179c88691ec4b8328cc2036dcdadf2')

    def test_5(self):
        kdf = lambda x: SHA256.new(x).digest()
        self.assertRaises(ValueError, key_agreement, kdf=kdf, static_priv=self.static_priv, eph_pub=self.eph_pub, eph_priv=self.eph_priv)

    def test_6(self):
        kdf = lambda x: SHA256.new(x).digest()
        self.assertRaises(ValueError, key_agreement, kdf=kdf, static_pub=self.static_pub, eph_pub=self.eph_pub, eph_priv=self.eph_priv)

    def test_7(self):
        kdf = lambda x: SHA256.new(x).digest()
        z = key_agreement(kdf=kdf, eph_pub=self.eph_pub, eph_priv=self.eph_priv)
        self.assertEqual(hexlify(z), b'feb257ebe063078b1391aac07913283d7b642ad7df61b46dfc9cd6f420bb896a')

    def test_8(self):
        kdf = lambda x: SHA256.new(x).digest()
        z = key_agreement(kdf=kdf, static_priv=self.static_priv, eph_pub=self.eph_pub)
        self.assertEqual(hexlify(z), b'ee4dc995117476ed57fd17ff0ed44e9f0466d46b929443bc0db9380317583b04')

    def test_9(self):
        kdf = lambda x: SHA256.new(x).digest()
        z = key_agreement(kdf=kdf, static_pub=self.static_pub, eph_priv=self.eph_priv)
        self.assertEqual(hexlify(z), b'2351cc2014f7c40468fa072b5d30f706eeaeef7507311cd8e59bab3b43f03c51')

    def test_10(self):
        kdf = lambda x: SHA256.new(x).digest()
        self.assertRaises(ValueError, key_agreement, kdf=kdf, static_pub=self.static_pub, eph_pub=self.eph_pub)

    def test_11(self):
        kdf = lambda x: SHA256.new(x).digest()
        self.assertRaises(ValueError, key_agreement, kdf=kdf, static_priv=self.static_priv, eph_priv=self.eph_priv)

    def test_12(self):
        self.assertRaises(ValueError, key_agreement, static_pub=self.static_pub, static_priv=self.static_priv)