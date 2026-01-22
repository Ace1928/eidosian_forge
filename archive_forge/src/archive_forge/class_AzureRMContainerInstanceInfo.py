from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
class AzureRMContainerInstanceInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.name = None
        self.tags = None
        super(AzureRMContainerInstanceInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_containerinstance_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_containerinstance_facts' module has been renamed to 'azure_rm_containerinstance_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['containerinstances'] = self.get()
        elif self.resource_group is not None:
            self.results['containerinstances'] = self.list_by_resource_group()
        else:
            self.results['containerinstances'] = self.list_all()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.containerinstance_client.container_groups.get(resource_group_name=self.resource_group, container_group_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Container Instances.')
        if response is not None and self.has_tags(response.tags, self.tags):
            results.append(self.format_item(response))
        return results

    def list_by_resource_group(self):
        response = None
        results = []
        try:
            response = self.containerinstance_client.container_groups.list_by_resource_group(resource_group_name=self.resource_group)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.fail('Could not list facts for Container Instances.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_item(item))
        return results

    def list_all(self):
        response = None
        results = []
        try:
            response = self.containerinstance_client.container_groups.list()
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.fail('Could not list facts for Container Instances.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_item(item))
        return results

    def format_item(self, item):
        d = item.as_dict()
        containers = d['containers']
        ports = d['ip_address']['ports'] if 'ip_address' in d else []
        resource_group = d['id'].split('resourceGroups/')[1].split('/')[0]
        for port_index in range(len(ports)):
            ports[port_index] = ports[port_index]['port']
        for container_index in range(len(containers)):
            old_container = containers[container_index]
            new_container = {'name': old_container['name'], 'image': old_container['image'], 'memory': old_container['resources']['requests']['memory_in_gb'], 'cpu': old_container['resources']['requests']['cpu'], 'ports': [], 'commands': old_container.get('command'), 'environment_variables': old_container.get('environment_variables'), 'volume_mounts': []}
            for port_index in range(len(old_container['ports'])):
                new_container['ports'].append(old_container['ports'][port_index]['port'])
            if 'volume_mounts' in old_container:
                for volume_mount_index in range(len(old_container['volume_mounts'])):
                    new_container['volume_mounts'].append(old_container['volume_mounts'][volume_mount_index])
            containers[container_index] = new_container
        d = {'id': d['id'], 'resource_group': resource_group, 'name': d['name'], 'os_type': d['os_type'], 'dns_name_label': d['ip_address'].get('dns_name_label'), 'ip_address': d['ip_address']['ip'] if 'ip_address' in d else '', 'ports': ports, 'location': d['location'], 'containers': containers, 'restart_policy': _camel_to_snake(d.get('restart_policy')) if d.get('restart_policy') else None, 'tags': d.get('tags', None), 'subnet_ids': d.get('subnet_ids', None), 'volumes': d['volumes'] if 'volumes' in d else []}
        return d