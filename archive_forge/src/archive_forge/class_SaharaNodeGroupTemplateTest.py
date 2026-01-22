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
class SaharaNodeGroupTemplateTest(common.HeatTestCase):

    def setUp(self):
        super(SaharaNodeGroupTemplateTest, self).setUp()
        self.stub_FlavorConstraint_validate()
        self.stub_SaharaPluginConstraint()
        self.stub_VolumeTypeConstraint_validate()
        self.patchobject(nova.NovaClientPlugin, 'find_flavor_by_name_or_id').return_value = 'someflavorid'
        self.patchobject(neutron.NeutronClientPlugin, '_create')
        self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', return_value='some_pool_id')
        sahara_mock = mock.MagicMock()
        self.ngt_mgr = sahara_mock.node_group_templates
        self.plugin_mgr = sahara_mock.plugins
        self.patchobject(sahara.SaharaClientPlugin, '_create').return_value = sahara_mock
        self.patchobject(sahara.SaharaClientPlugin, 'validate_hadoop_version').return_value = None
        self.fake_ngt = FakeNodeGroupTemplate()
        self.t = template_format.parse(node_group_template)
        self.ngt_props = self.t['resources']['node-group']['properties']

    def _init_ngt(self, template):
        self.stack = utils.parse_stack(template)
        return self.stack['node-group']

    def _create_ngt(self, template):
        ngt = self._init_ngt(template)
        self.ngt_mgr.create.return_value = self.fake_ngt
        scheduler.TaskRunner(ngt.create)()
        self.assertEqual((ngt.CREATE, ngt.COMPLETE), ngt.state)
        self.assertEqual(self.fake_ngt.id, ngt.resource_id)
        return ngt

    def test_ngt_create(self):
        self._create_ngt(self.t)
        args = {'name': 'node-group-template', 'plugin_name': 'vanilla', 'hadoop_version': '2.3.0', 'flavor_id': 'someflavorid', 'description': '', 'volumes_per_node': 0, 'volumes_size': None, 'volume_type': 'lvm', 'security_groups': None, 'auto_security_group': None, 'availability_zone': None, 'volumes_availability_zone': None, 'node_processes': ['namenode', 'jobtracker'], 'floating_ip_pool': 'some_pool_id', 'node_configs': None, 'image_id': None, 'is_proxy_gateway': True, 'volume_local_to_instance': None, 'use_autoconfig': None, 'shares': [{'id': 'e45eaabf-9300-42e2-b6eb-9ebc92081f46', 'access_level': 'ro', 'path': None}]}
        self.ngt_mgr.create.assert_called_once_with(**args)

    def test_validate_floatingippool_on_neutron_fails(self):
        ngt = self._init_ngt(self.t)
        self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id').side_effect = [neutron.exceptions.NeutronClientNoUniqueMatch(message='Too many'), neutron.exceptions.NeutronClientException(message='Not found', status_code=404)]
        ex = self.assertRaises(exception.StackValidationFailed, ngt.validate)
        self.assertEqual('Too many', str(ex))
        ex = self.assertRaises(exception.StackValidationFailed, ngt.validate)
        self.assertEqual('Not found', str(ex))

    def test_validate_flavor_constraint_return_false(self):
        self.t['resources']['node-group']['properties'].pop('floating_ip_pool')
        self.t['resources']['node-group']['properties'].pop('volume_type')
        ngt = self._init_ngt(self.t)
        self.patchobject(nova.FlavorConstraint, 'validate').return_value = False
        ex = self.assertRaises(exception.StackValidationFailed, ngt.validate)
        self.assertEqual(u"Property error: resources.node-group.properties.flavor: Error validating value 'm1.large'", str(ex))

    def test_template_invalid_name(self):
        tmpl = template_format.parse(node_group_template_without_name)
        stack = utils.parse_stack(tmpl)
        ngt = stack['node_group!']
        self.ngt_mgr.create.return_value = self.fake_ngt
        scheduler.TaskRunner(ngt.create)()
        self.assertEqual((ngt.CREATE, ngt.COMPLETE), ngt.state)
        self.assertEqual(self.fake_ngt.id, ngt.resource_id)
        name = self.ngt_mgr.create.call_args[1]['name']
        self.assertIn('-nodegroup-', name)

    def test_ngt_show_resource(self):
        ngt = self._create_ngt(self.t)
        self.ngt_mgr.get.return_value = self.fake_ngt
        self.assertEqual({'ng-template': 'info'}, ngt.FnGetAtt('show'))
        self.ngt_mgr.get.assert_called_once_with('some_ng_id')

    def test_validate_node_processes_fails(self):
        ngt = self._init_ngt(self.t)
        plugin_mock = mock.MagicMock()
        plugin_mock.node_processes = {'HDFS': ['namenode', 'datanode', 'secondarynamenode'], 'JobFlow': ['oozie']}
        self.plugin_mgr.get_version_details.return_value = plugin_mock
        ex = self.assertRaises(exception.StackValidationFailed, ngt.validate)
        self.assertIn("resources.node-group.properties: Plugin vanilla doesn't support the following node processes: jobtracker. Allowed processes are: ", str(ex))
        self.assertIn('namenode', str(ex))
        self.assertIn('datanode', str(ex))
        self.assertIn('secondarynamenode', str(ex))
        self.assertIn('oozie', str(ex))

    def test_update(self):
        ngt = self._create_ngt(self.t)
        props = self.ngt_props.copy()
        props['node_processes'] = ['tasktracker', 'datanode']
        props['name'] = 'new-ng-template'
        rsrc_defn = ngt.t.freeze(properties=props)
        scheduler.TaskRunner(ngt.update, rsrc_defn)()
        args = {'node_processes': ['tasktracker', 'datanode'], 'name': 'new-ng-template'}
        self.ngt_mgr.update.assert_called_once_with('some_ng_id', **args)
        self.assertEqual((ngt.UPDATE, ngt.COMPLETE), ngt.state)

    def test_get_live_state(self):
        ngt = self._create_ngt(self.t)
        resp = mock.MagicMock()
        resp.to_dict.return_value = {'volume_local_to_instance': False, 'availability_zone': None, 'updated_at': None, 'use_autoconfig': True, 'volumes_per_node': 0, 'id': '6157755e-dfd3-45b4-a445-36588e5f75ad', 'security_groups': None, 'shares': None, 'node_configs': {}, 'auto_security_group': False, 'volumes_availability_zone': None, 'description': '', 'volume_mount_prefix': '/volumes/disk', 'plugin_name': 'vanilla', 'floating_ip_pool': None, 'is_default': False, 'image_id': None, 'volumes_size': 0, 'is_proxy_gateway': False, 'is_public': False, 'hadoop_version': '2.7.1', 'name': 'cluster-nodetemplate-jlgzovdaivn', 'tenant_id': '221b4f51e9bd4f659845f657a3051a46', 'created_at': '2016-01-29T11:08:46', 'volume_type': None, 'is_protected': False, 'node_processes': ['namenode'], 'flavor_id': '2'}
        self.ngt_mgr.get.return_value = resp
        ngt.properties.data['flavor'] = '1'
        reality = ngt.get_live_state(ngt.properties)
        expected = {'volume_local_to_instance': False, 'availability_zone': None, 'use_autoconfig': True, 'volumes_per_node': 0, 'security_groups': None, 'shares': None, 'node_configs': {}, 'auto_security_group': False, 'volumes_availability_zone': None, 'description': '', 'plugin_name': 'vanilla', 'floating_ip_pool': None, 'image_id': None, 'volumes_size': 0, 'is_proxy_gateway': False, 'hadoop_version': '2.7.1', 'name': 'cluster-nodetemplate-jlgzovdaivn', 'volume_type': None, 'node_processes': ['namenode'], 'flavor': '2'}
        self.assertEqual(expected, reality)
        ngt.properties.data['flavor'] = '2'
        reality = ngt.get_live_state(ngt.properties)
        self.assertEqual('2', reality.get('flavor'))