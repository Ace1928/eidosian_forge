import hashlib
from oslo_utils.secretutils import md5
from oslotest import base
import glance_store.driver as driver
class TestBackCompatWrapper(base.BaseTestCase):

    def setUp(self):
        super(TestBackCompatWrapper, self).setUp()
        self.fake_store = _FakeStore()
        self.fake_context = _FakeContext()
        self.fake_verifier = _FakeVerifier()
        self.img_id = '1234'
        self.img_file = b'0123456789'
        self.img_size = 10
        self.img_checksum = md5(self.img_file, usedforsecurity=False).hexdigest()
        self.hashing_algo = 'sha256'
        self.img_sha256 = hashlib.sha256(self.img_file).hexdigest()

    def test_old_style_3_args(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size)
        self.assertEqual(tuple, type(x))
        self.assertEqual(4, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertIsInstance(x[3], dict)
        self.assertIsNone(x[3]['context_obj'])
        self.assertIsNone(x[3]['verifier_obj'])

    def test_old_style_4_args(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.fake_context)
        self.assertEqual(tuple, type(x))
        self.assertEqual(4, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertIsInstance(x[3], dict)
        self.assertEqual('context', x[3]['context_obj'])
        self.assertIsNone(x[3]['verifier_obj'])

    def test_old_style_5_args(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.fake_context, self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(4, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertIsInstance(x[3], dict)
        self.assertEqual('context', x[3]['context_obj'])
        self.assertEqual('verifier', x[3]['verifier_obj'])

    def test_old_style_3_args_kw_context(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, context=self.fake_context)
        self.assertEqual(tuple, type(x))
        self.assertEqual(4, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertIsInstance(x[3], dict)
        self.assertEqual('context', x[3]['context_obj'])
        self.assertIsNone(x[3]['verifier_obj'])

    def test_old_style_3_args_kw_verifier(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, verifier=self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(4, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertIsInstance(x[3], dict)
        self.assertIsNone(x[3]['context_obj'])
        self.assertEqual('verifier', x[3]['verifier_obj'])

    def test_old_style_4_args_kw_verifier(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.fake_context, verifier=self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(4, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertIsInstance(x[3], dict)
        self.assertEqual('context', x[3]['context_obj'])
        self.assertEqual('verifier', x[3]['verifier_obj'])

    def test_old_style_3_args_kws_context_verifier(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, context=self.fake_context, verifier=self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(4, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertIsInstance(x[3], dict)
        self.assertEqual('context', x[3]['context_obj'])
        self.assertEqual('verifier', x[3]['verifier_obj'])

    def test_old_style_all_kw_in_order(self):
        x = self.fake_store.add(image_id=self.img_id, image_file=self.img_file, image_size=self.img_size, context=self.fake_context, verifier=self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(4, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertIsInstance(x[3], dict)
        self.assertEqual('context', x[3]['context_obj'])
        self.assertEqual('verifier', x[3]['verifier_obj'])

    def test_old_style_all_kw_random_order(self):
        x = self.fake_store.add(image_file=self.img_file, context=self.fake_context, image_size=self.img_size, verifier=self.fake_verifier, image_id=self.img_id)
        self.assertEqual(tuple, type(x))
        self.assertEqual(4, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertIsInstance(x[3], dict)
        self.assertEqual('context', x[3]['context_obj'])
        self.assertEqual('verifier', x[3]['verifier_obj'])

    def test_new_style_6_args(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.hashing_algo, self.fake_context, self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertEqual('context', x[4]['context_obj'])
        self.assertEqual('verifier', x[4]['verifier_obj'])

    def test_new_style_3_args_kw_hash(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, hashing_algo=self.hashing_algo)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertIsNone(x[4]['context_obj'])
        self.assertIsNone(x[4]['verifier_obj'])

    def test_new_style_3_args_kws_context_hash(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, context=self.fake_context, hashing_algo=self.hashing_algo)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertEqual('context', x[4]['context_obj'])
        self.assertIsNone(x[4]['verifier_obj'])

    def test_new_style_3_args_kws_verifier_hash(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, hashing_algo=self.hashing_algo, verifier=self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertIsNone(x[4]['context_obj'])
        self.assertEqual('verifier', x[4]['verifier_obj'])

    def test_new_style_3_args_kws_hash_context_verifier(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, hashing_algo=self.hashing_algo, context=self.fake_context, verifier=self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertEqual('context', x[4]['context_obj'])
        self.assertEqual('verifier', x[4]['verifier_obj'])

    def test_new_style_4_args(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.hashing_algo)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertIsNone(x[4]['context_obj'])
        self.assertIsNone(x[4]['verifier_obj'])

    def test_new_style_4_args_kw_context(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.hashing_algo, context=self.fake_context)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertEqual('context', x[4]['context_obj'])
        self.assertIsNone(x[4]['verifier_obj'])

    def test_new_style_4_args_kws_verifier_context(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.hashing_algo, context=self.fake_context, verifier=self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertEqual('context', x[4]['context_obj'])
        self.assertEqual('verifier', x[4]['verifier_obj'])

    def test_new_style_5_args_kw_verifier(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.hashing_algo, self.fake_context, verifier=self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertEqual('context', x[4]['context_obj'])
        self.assertEqual('verifier', x[4]['verifier_obj'])

    def test_new_style_6_args_no_kw(self):
        x = self.fake_store.add(self.img_id, self.img_file, self.img_size, self.hashing_algo, self.fake_context, self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertEqual('context', x[4]['context_obj'])
        self.assertEqual('verifier', x[4]['verifier_obj'])

    def test_new_style_all_kw_in_order(self):
        x = self.fake_store.add(image_id=self.img_id, image_file=self.img_file, image_size=self.img_size, hashing_algo=self.hashing_algo, context=self.fake_context, verifier=self.fake_verifier)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertEqual('context', x[4]['context_obj'])
        self.assertEqual('verifier', x[4]['verifier_obj'])

    def test_new_style_all_kw_random_order(self):
        x = self.fake_store.add(hashing_algo=self.hashing_algo, image_file=self.img_file, context=self.fake_context, image_size=self.img_size, verifier=self.fake_verifier, image_id=self.img_id)
        self.assertEqual(tuple, type(x))
        self.assertEqual(5, len(x))
        self.assertIn(self.img_id, x[0])
        self.assertEqual(self.img_size, x[1])
        self.assertEqual(self.img_checksum, x[2])
        self.assertEqual(self.img_sha256, x[3])
        self.assertIsInstance(x[4], dict)
        self.assertEqual('context', x[4]['context_obj'])
        self.assertEqual('verifier', x[4]['verifier_obj'])

    def test_neg_too_few_args(self):
        self.assertRaises(TypeError, self.fake_store.add, self.img_id, self.img_file)

    def test_neg_too_few_kw_args(self):
        self.assertRaises(TypeError, self.fake_store.add, self.img_file, self.img_size, self.fake_context, self.fake_verifier, image_id=self.img_id)

    def test_neg_bogus_kw_args(self):
        self.assertRaises(TypeError, self.fake_store.add, thrashing_algo=self.hashing_algo, image_file=self.img_file, context=self.fake_context, image_size=self.img_size, verifier=self.fake_verifier, image_id=self.img_id)