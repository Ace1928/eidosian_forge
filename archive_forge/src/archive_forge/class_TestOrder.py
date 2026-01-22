from openstack.key_manager.v1 import order
from openstack.tests.unit import base
class TestOrder(base.TestCase):

    def test_basic(self):
        sot = order.Order()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('orders', sot.resources_key)
        self.assertEqual('/orders', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = order.Order(**EXAMPLE)
        self.assertEqual(EXAMPLE['created'], sot.created_at)
        self.assertEqual(EXAMPLE['creator_id'], sot.creator_id)
        self.assertEqual(EXAMPLE['meta'], sot.meta)
        self.assertEqual(EXAMPLE['order_ref'], sot.order_ref)
        self.assertEqual(ID_VAL, sot.order_id)
        self.assertEqual(EXAMPLE['secret_ref'], sot.secret_ref)
        self.assertEqual(SECRET_ID, sot.secret_id)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['sub_status'], sot.sub_status)
        self.assertEqual(EXAMPLE['sub_status_message'], sot.sub_status_message)
        self.assertEqual(EXAMPLE['type'], sot.type)
        self.assertEqual(EXAMPLE['updated'], sot.updated_at)