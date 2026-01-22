import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
class TestScryptHash(testm.TestCase):

    def setUp(self):
        self.input = 'message'
        self.password = 'password'
        self.salt = 'NaCl'
        self.hashes = []
        base_dir = dirname(abspath(__file__)) + sep
        hvf = open(base_dir + 'hashvectors.csv')
        hash_reader = reader(hvf, dialect='excel')
        for row in hash_reader:
            self.hashes.append(row)
        hvf.close()

    def test_hash_vectors_from_csv(self):
        """Test hash function with precalculated combinations."""
        for row in self.hashes[1:]:
            h = scrypt.hash(row[0], row[1], int(row[2]), int(row[3]), int(row[4]))
            hhex = b2a_hex(h)
            self.assertEqual(hhex, bytes(row[5].encode('utf-8')))

    def test_hash_buflen_keyword(self):
        """Test hash takes keyword valid buflen."""
        h64 = scrypt.hash(self.input, self.salt, buflen=64)
        h128 = scrypt.hash(self.input, self.salt, buflen=128)
        self.assertEqual(len(h64), 64)
        self.assertEqual(len(h128), 128)

    def test_hash_n_positional(self):
        """Test hash accepts valid N in position 3."""
        h = scrypt.hash(self.input, self.salt, 256)
        self.assertEqual(len(h), 64)

    def test_hash_n_keyword(self):
        """Test hash takes keyword valid N."""
        h = scrypt.hash(N=256, password=self.input, salt=self.salt)
        self.assertEqual(len(h), 64)

    def test_hash_r_positional(self):
        """Test hash accepts valid r in position 4."""
        h = scrypt.hash(self.input, self.salt, 256, 16)
        self.assertEqual(len(h), 64)

    def test_hash_r_keyword(self):
        """Test hash takes keyword valid r."""
        h = scrypt.hash(r=16, password=self.input, salt=self.salt)
        self.assertEqual(len(h), 64)

    def test_hash_p_positional(self):
        """Test hash accepts valid p in position 5."""
        h = scrypt.hash(self.input, self.salt, 256, 8, 2)
        self.assertEqual(len(h), 64)

    def test_hash_p_keyword(self):
        """Test hash takes keyword valid p."""
        h = scrypt.hash(p=4, password=self.input, salt=self.salt)
        self.assertEqual(len(h), 64)

    def test_hash_raises_error_on_p_equals_zero(self):
        """Test hash raises scrypt error on illegal parameter value (p = 0)"""
        self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, p=0))

    def test_hash_raises_error_on_negative_p(self):
        """Test hash raises scrypt error on illegal parameter value (p < 0)"""
        self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, p=-1))

    def test_hash_raises_error_on_r_equals_zero(self):
        """Test hash raises scrypt error on illegal parameter value (r = 0)"""
        self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, r=0))

    def test_hash_raises_error_on_negative_r(self):
        """Test hash raises scrypt error on illegal parameter value (r < 1)"""
        self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, r=-1))

    def test_hash_raises_error_r_p_over_limit(self):
        """Test hash raises scrypt error when parameters r multiplied by p over limit
        2**30."""
        self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, r=2, p=2 ** 29))

    def test_hash_raises_error_n_not_power_of_two(self):
        """Test hash raises scrypt error when parameter N is not a power of two {2, 4,
        8, 16, etc}"""
        self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, N=3))

    def test_hash_raises_error_n_under_limit(self):
        """Test hash raises scrypt error when parameter N under limit of 1."""
        self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, N=1))
        self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, N=-1))