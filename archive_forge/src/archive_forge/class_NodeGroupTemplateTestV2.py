from saharaclient.api import node_group_templates as ng
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
class NodeGroupTemplateTestV2(base.BaseTestCase):
    body = {'name': 'name', 'plugin_name': 'plugin', 'plugin_version': '1', 'flavor_id': '2', 'description': 'description', 'volumes_per_node': '3', 'volumes_size': '4', 'node_processes': ['datanode'], 'use_autoconfig': True, 'volume_mount_prefix': '/volumes/disk', 'boot_from_volume': False}
    update_json = {'node_group_template': {'name': 'UpdatedName', 'plugin_name': 'new_plugin', 'plugin_version': '2', 'flavor_id': '7', 'description': 'description', 'volumes_per_node': '3', 'volumes_size': '4', 'node_processes': ['datanode', 'namenode'], 'use_autoconfig': False, 'volume_mount_prefix': '/volumes/newdisk', 'boot_from_volume': True}}

    def test_create_node_group_template_v2(self):
        url = self.URL + '/node-group-templates'
        self.responses.post(url, status_code=202, json={'node_group_template': self.body})
        resp = self.client_v2.node_group_templates.create(**self.body)
        self.assertEqual(url, self.responses.last_request.url)
        self.assertEqual(self.body, json.loads(self.responses.last_request.body))
        self.assertIsInstance(resp, ng.NodeGroupTemplate)
        self.assertFields(self.body, resp)

    def test_update_node_group_template_v2(self):
        url = self.URL + '/node-group-templates'
        self.responses.post(url, status_code=202, json={'node_group_template': self.body})
        resp = self.client_v2.node_group_templates.create(**self.body)
        update_url = self.URL + '/node-group-templates/id'
        self.responses.patch(update_url, status_code=202, json=self.update_json)
        updated = self.client_v2.node_group_templates.update('id', resp.name, resp.plugin_name, resp.plugin_version, resp.flavor_id, description=getattr(resp, 'description', None), volumes_per_node=getattr(resp, 'volumes_per_node', None), node_configs=getattr(resp, 'node_configs', None), floating_ip_pool=getattr(resp, 'floating_ip_pool', None), security_groups=getattr(resp, 'security_groups', None), auto_security_group=getattr(resp, 'auto_security_group', None), availability_zone=getattr(resp, 'availability_zone', None), volumes_availability_zone=getattr(resp, 'volumes_availability_zone', None), volume_type=getattr(resp, 'volume_type', None), image_id=getattr(resp, 'image_id', None), is_proxy_gateway=getattr(resp, 'is_proxy_gateway', None), volume_local_to_instance=getattr(resp, 'volume_local_to_instance', None), use_autoconfig=False, boot_from_volume=getattr(resp, 'boot_from_volume', None))
        self.assertIsInstance(updated, ng.NodeGroupTemplate)
        self.assertFields(self.update_json['node_group_template'], updated)
        self.client_v2.node_group_templates.update('id')
        self.assertEqual(update_url, self.responses.last_request.url)
        self.assertEqual({}, json.loads(self.responses.last_request.body))
        unset_json = {'auto_security_group': None, 'availability_zone': None, 'description': None, 'flavor_id': None, 'floating_ip_pool': None, 'plugin_version': None, 'image_id': None, 'is_protected': None, 'is_proxy_gateway': None, 'is_public': None, 'name': None, 'node_configs': None, 'node_processes': None, 'plugin_name': None, 'security_groups': None, 'shares': None, 'use_autoconfig': None, 'volume_local_to_instance': None, 'volume_mount_prefix': None, 'volume_type': None, 'volumes_availability_zone': None, 'volumes_per_node': None, 'volumes_size': None, 'boot_from_volume': None}
        self.client_v2.node_group_templates.update('id', **unset_json)
        self.assertEqual(update_url, self.responses.last_request.url)
        self.assertEqual(unset_json, json.loads(self.responses.last_request.body))