from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMPrivateLinkServiceInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.name = None
        self.tags = None
        self.resource_group = None
        self.results = dict(changed=False)
        super(AzureRMPrivateLinkServiceInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        if self.name is not None and self.resource_group is not None:
            result = self.get_item()
        elif self.resource_group is not None:
            result = self.list_resourcegroup()
        else:
            result = self.list_by_subscription()
        self.results['link_service'] = [item for item in result if item and self.has_tags(item['tags'], self.tags)]
        return self.results

    def get_item(self):
        self.log('Get properties for {0} in {1}'.format(self.name, self.resource_group))
        try:
            response = self.network_client.private_link_services.get(self.resource_group, self.name)
            return [self.service_to_dict(response)]
        except ResourceNotFoundError:
            self.log('Could not get info for {0} in {1}'.format(self.name, self.resource_group))
        return []

    def list_resourcegroup(self):
        result = []
        self.log('List all in {0}'.format(self.resource_group))
        try:
            response = self.network_client.private_link_services.list(self.resource_group)
            while True:
                result.append(response.next())
        except StopIteration:
            pass
        except Exception:
            pass
        return [self.service_to_dict(item) for item in result]

    def list_by_subscription(self):
        result = []
        self.log('List all in by subscription')
        try:
            response = self.network_client.private_link_services.list_by_subscription()
            while True:
                result.append(response.next())
        except StopIteration:
            pass
        except Exception:
            pass
        return [self.service_to_dict(item) for item in result]

    def service_to_dict(self, service_info):
        service = service_info.as_dict()
        result = dict(id=service.get('id'), name=service.get('name'), type=service.get('type'), etag=service.get('etag'), location=service.get('location'), tags=service.get('tags'), load_balancer_frontend_ip_configurations=service.get('load_balancer_frontend_ip_configurations'), ip_configurations=list(), network_interfaces=service.get('network_interfaces'), provisioning_state=service.get('provisioning_state'), private_endpoint_connections=list(), visibility=service.get('visibility'), auto_approval=service.get('auto_approval'), fqdns=service.get('fqdns'), alias=service.get('alias'), enable_proxy_protocol=service.get('enable_proxy_protocol'))
        if service.get('private_endpoint_connections'):
            for items in service['private_endpoint_connections']:
                result['private_endpoint_connections'].append({'id': items['id'], 'private_endpoint': items['private_endpoint']['id']})
        if service.get('ip_configurations'):
            for items in service['ip_configurations']:
                result['ip_configurations'].append({'name': items['name'], 'properties': {'primary': items['primary'], 'private_ip_address_version': items['private_ip_address_version'], 'private_ip_allocation_method': items['private_ip_allocation_method'], 'subnet': items['subnet']}})
        return result