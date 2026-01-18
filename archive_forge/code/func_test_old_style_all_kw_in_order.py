import hashlib
from oslo_utils.secretutils import md5
from oslotest import base
import glance_store.driver as driver
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