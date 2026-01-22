from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMBastionHostInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.name = None
        self.tags = None
        self.resource_group = None
        self.results = dict(changed=False)
        super(AzureRMBastionHostInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        if self.name is not None and self.resource_group is not None:
            result = self.get_item()
        elif self.resource_group is not None:
            result = self.list_resourcegroup()
        else:
            result = self.list_by_subscription()
        self.results['bastion_host'] = [item for item in result if item and self.has_tags(item['tags'], self.tags)]
        return self.results

    def get_item(self):
        self.log('Get properties for {0} in {1}'.format(self.name, self.resource_group))
        try:
            response = self.network_client.bastion_hosts.get(self.resource_group, self.name)
            return [self.bastion_to_dict(response)]
        except ResourceNotFoundError:
            self.log('Could not get info for {0} in {1}'.format(self.name, self.resource_group))
        return []

    def list_resourcegroup(self):
        result = []
        self.log('List all in {0}'.format(self.resource_group))
        try:
            response = self.network_client.bastion_hosts.list_by_resource_group(self.resource_group)
            while True:
                result.append(response.next())
        except StopIteration:
            pass
        except Exception:
            pass
        return [self.bastion_to_dict(item) for item in result]

    def list_by_subscription(self):
        result = []
        self.log('List all in by subscription')
        try:
            response = self.network_client.bastion_hosts.list()
            while True:
                result.append(response.next())
        except StopIteration:
            pass
        except Exception:
            pass
        return [self.bastion_to_dict(item) for item in result]

    def bastion_to_dict(self, bastion_info):
        bastion = bastion_info.as_dict()
        result = dict(id=bastion.get('id'), name=bastion.get('name'), type=bastion.get('type'), etag=bastion.get('etag'), location=bastion.get('location'), tags=bastion.get('tags'), sku=dict(), ip_configurations=list(), dns_name=bastion.get('dns_name'), provisioning_state=bastion.get('provisioning_state'), scale_units=bastion.get('scale_units'), disable_copy_paste=bastion.get('disable_copy_paste'), enable_file_copy=bastion.get('enable_file_copy'), enable_ip_connect=bastion.get('enable_ip_connect'), enable_shareable_link=bastion.get('enable_tunneling'), enable_tunneling=bastion.get('enable_tunneling'))
        if bastion.get('sku'):
            result['sku']['name'] = bastion['sku']['name']
        if bastion.get('ip_configurations'):
            for items in bastion['ip_configurations']:
                result['ip_configurations'].append({'name': items['name'], 'subnet': dict(id=items['subnet']['id']), 'public_ip_address': dict(id=items['public_ip_address']['id']), 'private_ip_allocation_method': items['private_ip_allocation_method']})
        return result