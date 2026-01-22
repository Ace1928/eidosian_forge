import copy
from unittest import mock
from neutronclient.neutron import v2_0 as neutronV20
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.magnum import cluster_template
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class TestMagnumClusterTemplate(common.HeatTestCase):
    magnum_template = "\n    heat_template_version: ocata\n    resources:\n      test_cluster_template:\n        type: OS::Magnum::ClusterTemplate\n        properties:\n          name: test_cluster_template\n          image: fedora-21-atomic-2\n          flavor: m1.small\n          master_flavor: m1.medium\n          keypair: heat_key\n          external_network: 0244b54d-ae1f-44f0-a24a-442760f1d681\n          fixed_network: 0f59a3dd-fac1-4d03-b41a-d4115fbffa89\n          fixed_subnet: 27a8c89c-0d28-4946-8c78-82cfec1d670a\n          dns_nameserver: 8.8.8.8\n          docker_volume_size: 5\n          docker_storage_driver: devicemapper\n          coe: 'mesos'\n          network_driver: 'flannel'\n          http_proxy: 'http://proxy.com:123'\n          https_proxy: 'https://proxy.com:123'\n          no_proxy: '192.168.0.1'\n          labels: {'flannel_cidr': ['10.101.0.0/16', '10.102.0.0/16']}\n          tls_disabled: True\n          public: True\n          registry_enabled: True\n          volume_driver: rexray\n          server_type: vm\n          master_lb_enabled: True\n          floating_ip_enabled: True\n    "
    expected = {'name': 'test_cluster_template', 'image_id': 'fedora-21-atomic-2', 'flavor_id': 'm1.small', 'master_flavor_id': 'm1.medium', 'keypair_id': 'heat_key', 'external_network_id': 'id_for_net_or_sub', 'fixed_network': 'id_for_net_or_sub', 'fixed_subnet': 'id_for_net_or_sub', 'dns_nameserver': '8.8.8.8', 'docker_volume_size': 5, 'docker_storage_driver': 'devicemapper', 'coe': 'mesos', 'network_driver': 'flannel', 'http_proxy': 'http://proxy.com:123', 'https_proxy': 'https://proxy.com:123', 'no_proxy': '192.168.0.1', 'labels': {'flannel_cidr': ['10.101.0.0/16', '10.102.0.0/16']}, 'tls_disabled': True, 'public': True, 'registry_enabled': True, 'volume_driver': 'rexray', 'server_type': 'vm', 'master_lb_enabled': True, 'floating_ip_enabled': True}

    def setUp(self):
        super(TestMagnumClusterTemplate, self).setUp()
        resource._register_class(RESOURCE_TYPE, cluster_template.ClusterTemplate)
        self.t = template_format.parse(self.magnum_template)
        self.stack = utils.parse_stack(self.t)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        self.rsrc_defn = resource_defns['test_cluster_template']
        self.client = mock.Mock()
        self.patchobject(cluster_template.ClusterTemplate, 'client', return_value=self.client)
        self.find_mock = self.patchobject(neutronV20, 'find_resourceid_by_name_or_id')
        self.find_mock.return_value = 'id_for_net_or_sub'
        self.stub_FlavorConstraint_validate()
        self.stub_KeypairConstraint_validate()
        self.stub_ImageConstraint_validate()
        self.stub_NetworkConstraint_validate()
        self.stub_SubnetConstraint_validate()

    def _create_resource(self, name, snippet, stack):
        self.resource_id = '12345'
        self.test_cluster_template = self.stack['test_cluster_template']
        value = mock.MagicMock(uuid=self.resource_id)
        self.client.cluster_templates.create.return_value = value
        bm = cluster_template.ClusterTemplate(name, snippet, stack)
        scheduler.TaskRunner(bm.create)()
        return bm

    def test_cluster_template_create(self):
        bm = self._create_resource('bm', self.rsrc_defn, self.stack)
        self.assertEqual(self.resource_id, bm.resource_id)
        self.assertEqual((bm.CREATE, bm.COMPLETE), bm.state)
        self.client.cluster_templates.create.assert_called_once_with(**self.expected)

    def test_validate_invalid_volume_driver(self):
        props = self.t['resources']['test_cluster_template']['properties']
        props['volume_driver'] = 'cinder'
        stack = utils.parse_stack(self.t)
        msg = "Volume driver type cinder is not supported by COE:mesos, expecting a ['rexray'] volume driver."
        ex = self.assertRaises(exception.StackValidationFailed, stack['test_cluster_template'].validate)
        self.assertEqual(msg, str(ex))

    def _cluster_template_update(self, update_status='UPDATE_COMPLETE', exc_msg=None):
        ct = self._create_resource('ct', self.rsrc_defn, self.stack)
        status = mock.MagicMock(status=update_status)
        self.client.cluster_templates.get.return_value = status
        t = template_format.parse(self.magnum_template)
        new_t = copy.deepcopy(t)
        new_t['resources'][self.expected['name']]['properties'][cluster_template.ClusterTemplate.PUBLIC] = False
        rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
        new_ct = rsrc_defns[self.expected['name']]
        if update_status == 'UPDATE_COMPLETE':
            scheduler.TaskRunner(ct.update, new_ct)()
            self.assertEqual((ct.UPDATE, ct.COMPLETE), ct.state)
        else:
            exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(ct.update, new_ct))
            self.assertIn(exc_msg, str(exc))

    def test_cluster_update(self):
        self._cluster_template_update()

    def test_cluster_update_failed(self):
        self._cluster_template_update('UPDATE_FAILED', 'Failed to update Cluster')

    def test_cluster_update_unknown_status(self):
        self._cluster_template_update('UPDATE_BAR', 'Unknown status updating Cluster')