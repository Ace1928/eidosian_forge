from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
class TestBaremetalPort(base.IronicTestCase):

    def setUp(self):
        super(TestBaremetalPort, self).setUp()
        self.fake_baremetal_node = fakes.make_fake_machine(self.name, self.uuid)
        self.fake_baremetal_port = fakes.make_fake_port('00:01:02:03:04:05', node_id=self.uuid)
        self.fake_baremetal_port2 = fakes.make_fake_port('0a:0b:0c:0d:0e:0f', node_id=self.uuid)

    def test_list_nics(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail']), json={'ports': [self.fake_baremetal_port, self.fake_baremetal_port2]})])
        return_value = self.cloud.list_nics()
        self.assertEqual(2, len(return_value))
        self.assertSubdict(self.fake_baremetal_port, return_value[0])
        self.assert_calls()

    def test_list_nics_failure(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail']), status_code=400)])
        self.assertRaises(exceptions.SDKException, self.cloud.list_nics)
        self.assert_calls()

    def test_list_nics_for_machine(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail'], qs_elements=['node_uuid=%s' % self.fake_baremetal_node['uuid']]), json={'ports': [self.fake_baremetal_port, self.fake_baremetal_port2]})])
        return_value = self.cloud.list_nics_for_machine(self.fake_baremetal_node['uuid'])
        self.assertEqual(2, len(return_value))
        self.assertSubdict(self.fake_baremetal_port, return_value[0])
        self.assert_calls()

    def test_list_nics_for_machine_failure(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail'], qs_elements=['node_uuid=%s' % self.fake_baremetal_node['uuid']]), status_code=400)])
        self.assertRaises(exceptions.SDKException, self.cloud.list_nics_for_machine, self.fake_baremetal_node['uuid'])
        self.assert_calls()

    def test_get_nic_by_mac(self):
        mac = self.fake_baremetal_port['address']
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail'], qs_elements=['address=%s' % mac]), json={'ports': [self.fake_baremetal_port]})])
        return_value = self.cloud.get_nic_by_mac(mac)
        self.assertSubdict(self.fake_baremetal_port, return_value)
        self.assert_calls()

    def test_get_nic_by_mac_failure(self):
        mac = self.fake_baremetal_port['address']
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail'], qs_elements=['address=%s' % mac]), json={'ports': []})])
        self.assertIsNone(self.cloud.get_nic_by_mac(mac))
        self.assert_calls()