import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import hypervisor
from openstack import exceptions
from openstack.tests.unit import base
class TestHypervisor(base.TestCase):

    def setUp(self):
        super(TestHypervisor, self).setUp()
        self.sess = mock.Mock(spec=adapter.Adapter)
        self.sess.default_microversion = 1
        self.sess._get_connection = mock.Mock(return_value=self.cloud)

    def test_basic(self):
        sot = hypervisor.Hypervisor()
        self.assertEqual('hypervisor', sot.resource_key)
        self.assertEqual('hypervisors', sot.resources_key)
        self.assertEqual('/os-hypervisors', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'hypervisor_hostname_pattern': 'hypervisor_hostname_pattern', 'limit': 'limit', 'marker': 'marker', 'with_servers': 'with_servers'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = hypervisor.Hypervisor(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['cpu_info'], sot.cpu_info)
        self.assertEqual(EXAMPLE['host_ip'], sot.host_ip)
        self.assertEqual(EXAMPLE['hypervisor_type'], sot.hypervisor_type)
        self.assertEqual(EXAMPLE['hypervisor_version'], sot.hypervisor_version)
        self.assertEqual(EXAMPLE['hypervisor_hostname'], sot.name)
        self.assertEqual(EXAMPLE['service'], sot.service_details)
        self.assertEqual(EXAMPLE['servers'], sot.servers)
        self.assertEqual(EXAMPLE['state'], sot.state)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['uptime'], sot.uptime)
        self.assertEqual(EXAMPLE['vcpus_used'], sot.vcpus_used)
        self.assertEqual(EXAMPLE['hypervisor_type'], sot.hypervisor_type)
        self.assertEqual(EXAMPLE['local_gb_used'], sot.local_disk_used)
        self.assertEqual(EXAMPLE['vcpus'], sot.vcpus)
        self.assertEqual(EXAMPLE['vcpus_used'], sot.vcpus_used)
        self.assertEqual(EXAMPLE['memory_mb_used'], sot.memory_used)
        self.assertEqual(EXAMPLE['memory_mb'], sot.memory_size)
        self.assertEqual(EXAMPLE['current_workload'], sot.current_workload)
        self.assertEqual(EXAMPLE['running_vms'], sot.running_vms)
        self.assertEqual(EXAMPLE['free_disk_gb'], sot.local_disk_free)
        self.assertEqual(EXAMPLE['disk_available_least'], sot.disk_available)
        self.assertEqual(EXAMPLE['local_gb'], sot.local_disk_size)
        self.assertEqual(EXAMPLE['free_ram_mb'], sot.memory_free)

    @mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=False)
    def test_get_uptime(self, mv_mock):
        sot = hypervisor.Hypervisor(**copy.deepcopy(EXAMPLE))
        rsp = {'hypervisor': {'hypervisor_hostname': 'fake-mini', 'id': sot.id, 'state': 'up', 'status': 'enabled', 'uptime': '08:32:11 up 93 days, 18:25, 12 users'}}
        resp = mock.Mock()
        resp.body = copy.deepcopy(rsp)
        resp.json = mock.Mock(return_value=resp.body)
        resp.headers = {}
        resp.status_code = 200
        self.sess.get = mock.Mock(return_value=resp)
        hyp = sot.get_uptime(self.sess)
        self.sess.get.assert_called_with('os-hypervisors/{id}/uptime'.format(id=sot.id), microversion=self.sess.default_microversion)
        self.assertEqual(rsp['hypervisor']['uptime'], hyp.uptime)
        self.assertEqual(rsp['hypervisor']['status'], sot.status)

    @mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=True)
    def test_get_uptime_after_2_88(self, mv_mock):
        sot = hypervisor.Hypervisor(**copy.deepcopy(EXAMPLE))
        self.assertRaises(exceptions.SDKException, sot.get_uptime, self.sess)