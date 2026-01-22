import os
import errno
import warnings
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import bord, tostr, FileNotFoundError
from Cryptodome.Util.asn1 import DerSequence, DerBitString
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Hash import SHAKE128
from Cryptodome.PublicKey import ECC
class TestExport_P192(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestExport_P192, self).__init__(*args, **kwargs)
        self.ref_private, self.ref_public = create_ref_keys_p192()

    def test_export_public_der_uncompressed(self):
        key_file = load_file('ecc_p192_public.der')
        encoded = self.ref_public._export_subjectPublicKeyInfo(False)
        self.assertEqual(key_file, encoded)
        encoded = self.ref_public.export_key(format='DER')
        self.assertEqual(key_file, encoded)
        encoded = self.ref_public.export_key(format='DER', compress=False)
        self.assertEqual(key_file, encoded)

    def test_export_public_der_compressed(self):
        key_file = load_file('ecc_p192_public.der')
        pub_key = ECC.import_key(key_file)
        key_file_compressed = pub_key.export_key(format='DER', compress=True)
        key_file_compressed_ref = load_file('ecc_p192_public_compressed.der')
        self.assertEqual(key_file_compressed, key_file_compressed_ref)

    def test_export_public_sec1_uncompressed(self):
        key_file = load_file('ecc_p192_public.der')
        value = extract_bitstring_from_spki(key_file)
        encoded = self.ref_public.export_key(format='SEC1')
        self.assertEqual(value, encoded)

    def test_export_public_sec1_compressed(self):
        key_file = load_file('ecc_p192_public.der')
        encoded = self.ref_public.export_key(format='SEC1', compress=True)
        key_file_compressed_ref = load_file('ecc_p192_public_compressed.der')
        value = extract_bitstring_from_spki(key_file_compressed_ref)
        self.assertEqual(value, encoded)

    def test_export_rfc5915_private_der(self):
        key_file = load_file('ecc_p192_private.der')
        encoded = self.ref_private._export_rfc5915_private_der()
        self.assertEqual(key_file, encoded)
        encoded = self.ref_private.export_key(format='DER', use_pkcs8=False)
        self.assertEqual(key_file, encoded)

    def test_export_private_pkcs8_clear(self):
        key_file = load_file('ecc_p192_private_p8_clear.der')
        encoded = self.ref_private._export_pkcs8()
        self.assertEqual(key_file, encoded)
        encoded = self.ref_private.export_key(format='DER')
        self.assertEqual(key_file, encoded)

    def test_export_private_pkcs8_encrypted(self):
        encoded = self.ref_private._export_pkcs8(passphrase='secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
        self.assertRaises(ValueError, ECC._import_pkcs8, encoded, None)
        decoded = ECC._import_pkcs8(encoded, 'secret')
        self.assertEqual(self.ref_private, decoded)
        encoded = self.ref_private.export_key(format='DER', passphrase='secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
        decoded = ECC.import_key(encoded, 'secret')
        self.assertEqual(self.ref_private, decoded)
        encoded = self.ref_private.export_key(format='DER', passphrase='secret', protection='PBKDF2WithHMAC-SHA224AndAES192-CBC', prot_params={'iteration_count': 123})
        decoded = ECC.import_key(encoded, 'secret')
        self.assertEqual(self.ref_private, decoded)

    def test_export_public_pem_uncompressed(self):
        key_file = load_file('ecc_p192_public.pem', 'rt').strip()
        encoded = self.ref_private._export_public_pem(False)
        self.assertEqual(key_file, encoded)
        encoded = self.ref_public.export_key(format='PEM')
        self.assertEqual(key_file, encoded)
        encoded = self.ref_public.export_key(format='PEM', compress=False)
        self.assertEqual(key_file, encoded)

    def test_export_public_pem_compressed(self):
        key_file = load_file('ecc_p192_public.pem', 'rt').strip()
        pub_key = ECC.import_key(key_file)
        key_file_compressed = pub_key.export_key(format='PEM', compress=True)
        key_file_compressed_ref = load_file('ecc_p192_public_compressed.pem', 'rt').strip()
        self.assertEqual(key_file_compressed, key_file_compressed_ref)

    def test_export_private_pem_clear(self):
        key_file = load_file('ecc_p192_private.pem', 'rt').strip()
        encoded = self.ref_private._export_private_pem(None)
        self.assertEqual(key_file, encoded)
        encoded = self.ref_private.export_key(format='PEM', use_pkcs8=False)
        self.assertEqual(key_file, encoded)

    def test_export_private_pem_encrypted(self):
        encoded = self.ref_private._export_private_pem(passphrase=b'secret')
        self.assertRaises(ValueError, ECC.import_key, encoded)
        assert 'EC PRIVATE KEY' in encoded
        decoded = ECC.import_key(encoded, 'secret')
        self.assertEqual(self.ref_private, decoded)
        encoded = self.ref_private.export_key(format='PEM', passphrase='secret', use_pkcs8=False)
        decoded = ECC.import_key(encoded, 'secret')
        self.assertEqual(self.ref_private, decoded)

    def test_export_private_pkcs8_and_pem_1(self):
        key_file = load_file('ecc_p192_private_p8_clear.pem', 'rt').strip()
        encoded = self.ref_private._export_private_clear_pkcs8_in_clear_pem()
        self.assertEqual(key_file, encoded)
        encoded = self.ref_private.export_key(format='PEM')
        self.assertEqual(key_file, encoded)

    def test_export_private_pkcs8_and_pem_2(self):
        encoded = self.ref_private._export_private_encrypted_pkcs8_in_clear_pem('secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
        self.assertRaises(ValueError, ECC.import_key, encoded)
        assert 'ENCRYPTED PRIVATE KEY' in encoded
        decoded = ECC.import_key(encoded, 'secret')
        self.assertEqual(self.ref_private, decoded)
        encoded = self.ref_private.export_key(format='PEM', passphrase='secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
        decoded = ECC.import_key(encoded, 'secret')
        self.assertEqual(self.ref_private, decoded)

    def test_prng(self):
        encoded1 = self.ref_private.export_key(format='PEM', passphrase='secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC', randfunc=get_fixed_prng())
        encoded2 = self.ref_private.export_key(format='PEM', passphrase='secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC', randfunc=get_fixed_prng())
        self.assertEqual(encoded1, encoded2)
        encoded1 = self.ref_private.export_key(format='PEM', use_pkcs8=False, passphrase='secret', randfunc=get_fixed_prng())
        encoded2 = self.ref_private.export_key(format='PEM', use_pkcs8=False, passphrase='secret', randfunc=get_fixed_prng())
        self.assertEqual(encoded1, encoded2)

    def test_byte_or_string_passphrase(self):
        encoded1 = self.ref_private.export_key(format='PEM', use_pkcs8=False, passphrase='secret', randfunc=get_fixed_prng())
        encoded2 = self.ref_private.export_key(format='PEM', use_pkcs8=False, passphrase=b'secret', randfunc=get_fixed_prng())
        self.assertEqual(encoded1, encoded2)

    def test_error_params1(self):
        self.assertRaises(ValueError, self.ref_private.export_key, format='XXX')
        self.ref_private.export_key(format='PEM', passphrase='secret', use_pkcs8=False)
        self.assertRaises(ValueError, self.ref_private.export_key, format='PEM', passphrase='secret')
        self.assertRaises(ValueError, self.ref_private.export_key, format='DER', passphrase='secret', use_pkcs8=False, protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
        self.assertRaises(ValueError, self.ref_public.export_key, format='DER', use_pkcs8=False)
        self.assertRaises(ValueError, self.ref_private.export_key, format='PEM', passphrase='', use_pkcs8=False)
        self.assertRaises(ValueError, self.ref_private.export_key, format='PEM', passphrase='', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')

    def test_compressed_curve(self):
        pem1 = '-----BEGIN EC PRIVATE KEY-----\n        MF8CAQEEGHvhXmIW95JxZYfd4AUPu9BwknjuvS36aqAKBggqhkjOPQMBAaE0AzIA\n        BLJZCyTu35DQIlqvMlBynn3k1Ig+dWfg/brRhHecxptrbloqFSP8ITw0CwbGF+2X\n        5g==\n        -----END EC PRIVATE KEY-----'
        pem2 = '-----BEGIN EC PRIVATE KEY-----\n        MF8CAQEEGA3rAotUaWl7d47eX6tz9JmLzOMJwl13XaAKBggqhkjOPQMBAaE0AzIA\n        BG4tHlTBBBGokcWmGm2xubVB0NvPC/Ou5AYwivs+3iCxmEjsymVAj6iiuX2Lxr6g\n        /Q==\n        -----END EC PRIVATE KEY-----'
        key1 = ECC.import_key(pem1)
        low16 = int(key1.pointQ.y % 65536)
        self.assertEqual(low16, 38886)
        key2 = ECC.import_key(pem2)
        low16 = int(key2.pointQ.y % 65536)
        self.assertEqual(low16, 41213)