from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class SaharaClusterTemplateTest(common.HeatTestCase):

    def setUp(self):
        super(SaharaClusterTemplateTest, self).setUp()
        self.patchobject(st.constraints.CustomConstraint, '_is_valid').return_value = True
        self.patchobject(neutron.NeutronClientPlugin, '_create')
        self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', return_value='some_network_id')
        sahara_mock = mock.MagicMock()
        self.ct_mgr = sahara_mock.cluster_templates
        self.patchobject(sahara.SaharaClientPlugin, '_create').return_value = sahara_mock
        self.patchobject(sahara.SaharaClientPlugin, 'validate_hadoop_version').return_value = None
        self.fake_ct = FakeClusterTemplate()
        self.t = template_format.parse(cluster_template)

    def _init_ct(self, template):
        self.stack = utils.parse_stack(template)
        return self.stack['cluster-template']

    def _create_ct(self, template):
        ct = self._init_ct(template)
        self.ct_mgr.create.return_value = self.fake_ct
        scheduler.TaskRunner(ct.create)()
        self.assertEqual((ct.CREATE, ct.COMPLETE), ct.state)
        self.assertEqual(self.fake_ct.id, ct.resource_id)
        return ct

    def test_ct_create(self):
        self._create_ct(self.t)
        args = {'name': 'test-cluster-template', 'plugin_name': 'vanilla', 'hadoop_version': '2.3.0', 'description': '', 'default_image_id': None, 'net_id': 'some_network_id', 'anti_affinity': None, 'node_groups': None, 'cluster_configs': None, 'use_autoconfig': None, 'shares': [{'id': 'e45eaabf-9300-42e2-b6eb-9ebc92081f46', 'access_level': 'ro', 'path': None}]}
        self.ct_mgr.create.assert_called_once_with(**args)

    def test_ct_validate_no_network_on_neutron_fails(self):
        self.t['resources']['cluster-template']['properties'].pop('neutron_management_network')
        ct = self._init_ct(self.t)
        self.patchobject(ct, 'is_using_neutron', return_value=True)
        ex = self.assertRaises(exception.StackValidationFailed, ct.validate)
        self.assertEqual('neutron_management_network must be provided', str(ex))

    def test_template_invalid_name(self):
        tmpl = template_format.parse(cluster_template_without_name)
        stack = utils.parse_stack(tmpl)
        ct = stack['cluster_template!']
        self.ct_mgr.create.return_value = self.fake_ct
        scheduler.TaskRunner(ct.create)()
        self.assertEqual((ct.CREATE, ct.COMPLETE), ct.state)
        self.assertEqual(self.fake_ct.id, ct.resource_id)
        name = self.ct_mgr.create.call_args[1]['name']
        self.assertIn('-clustertemplate-', name)

    def test_ct_show_resource(self):
        ct = self._create_ct(self.t)
        self.ct_mgr.get.return_value = self.fake_ct
        self.assertEqual({'cluster-template': 'info'}, ct.FnGetAtt('show'))
        self.ct_mgr.get.assert_called_once_with('some_ct_id')

    def test_update(self):
        ct = self._create_ct(self.t)
        rsrc_defn = self.stack.t.resource_definitions(self.stack)['cluster-template']
        props = self.t['resources']['cluster-template']['properties'].copy()
        props['plugin_name'] = 'hdp'
        props['hadoop_version'] = '1.3.2'
        props['name'] = 'new-cluster-template'
        rsrc_defn = rsrc_defn.freeze(properties=props)
        scheduler.TaskRunner(ct.update, rsrc_defn)()
        args = {'plugin_name': 'hdp', 'hadoop_version': '1.3.2', 'name': 'new-cluster-template'}
        self.ct_mgr.update.assert_called_once_with('some_ct_id', **args)
        self.assertEqual((ct.UPDATE, ct.COMPLETE), ct.state)

    def test_ct_get_live_state(self):
        ct = self._create_ct(self.t)
        resp = mock.MagicMock()
        resp.to_dict.return_value = {'neutron_management_network': 'public', 'description': '', 'cluster_configs': {}, 'created_at': '2016-01-29T11:45:47', 'default_image_id': None, 'updated_at': None, 'plugin_name': 'vanilla', 'shares': None, 'is_default': False, 'is_protected': False, 'use_autoconfig': True, 'anti_affinity': [], 'tenant_id': '221b4f51e9bd4f659845f657a3051a46', 'node_groups': [{'volume_local_to_instance': False, 'availability_zone': None, 'updated_at': None, 'node_group_template_id': '1234', 'volumes_per_node': 0, 'id': '48c356f6-bbe1-4b26-a90a-f3d543c2ea4c', 'security_groups': None, 'shares': None, 'node_configs': {}, 'auto_security_group': False, 'volumes_availability_zone': None, 'volume_mount_prefix': '/volumes/disk', 'floating_ip_pool': None, 'image_id': None, 'volumes_size': 0, 'is_proxy_gateway': False, 'count': 1, 'name': 'test', 'created_at': '2016-01-29T11:45:47', 'volume_type': None, 'node_processes': ['namenode'], 'flavor_id': '2', 'use_autoconfig': True}], 'is_public': False, 'hadoop_version': '2.7.1', 'id': 'c07b8c63-b944-47f9-8588-085547a45c1b', 'name': 'cluster-template-ykokor6auha4'}
        self.ct_mgr.get.return_value = resp
        reality = ct.get_live_state(ct.properties)
        expected = {'neutron_management_network': 'public', 'description': '', 'cluster_configs': {}, 'default_image_id': None, 'plugin_name': 'vanilla', 'shares': None, 'anti_affinity': [], 'node_groups': [{'node_group_template_id': '1234', 'count': 1, 'name': 'test'}], 'hadoop_version': '2.7.1', 'name': 'cluster-template-ykokor6auha4'}
        self.assertEqual(set(expected.keys()), set(reality.keys()))
        expected_node_group = sorted(expected.pop('node_groups'))
        reality_node_group = sorted(reality.pop('node_groups'))
        for i in range(len(expected_node_group)):
            self.assertEqual(expected_node_group[i], reality_node_group[i])
        self.assertEqual(expected, reality)