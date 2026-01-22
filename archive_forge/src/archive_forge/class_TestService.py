from unittest import mock
from openstack.clustering.v1 import service
from openstack.tests.unit import base
class TestService(base.TestCase):

    def setUp(self):
        super(TestService, self).setUp()
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.sess = mock.Mock()
        self.sess.put = mock.Mock(return_value=self.resp)

    def test_basic(self):
        sot = service.Service()
        self.assertEqual('service', sot.resource_key)
        self.assertEqual('services', sot.resources_key)
        self.assertEqual('/services', sot.base_path)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = service.Service(**EXAMPLE)
        self.assertEqual(EXAMPLE['host'], sot.host)
        self.assertEqual(EXAMPLE['binary'], sot.binary)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['state'], sot.state)
        self.assertEqual(EXAMPLE['disabled_reason'], sot.disabled_reason)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)