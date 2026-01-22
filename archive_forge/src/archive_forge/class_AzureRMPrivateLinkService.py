from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMPrivateLinkService(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', required=True), resource_group=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), load_balancer_frontend_ip_configurations=dict(type='list', elements='dict', options=load_balancer_frontend_ip_configurations_spec), ip_configurations=dict(type='list', elements='dict', options=dict(name=dict(type='str'), properties=dict(type='dict', options=properties_spec))), visibility=dict(type='dict', options=visibility_spec), fqdns=dict(type='list', elements='str'), enable_proxy_protocol=dict(type='bool'), auto_approval=dict(type='dict', options=auto_approval_spec))
        self.name = None
        self.resource_group = None
        self.location = None
        self.tags = None
        self.state = None
        self.results = dict(changed=False)
        self.body = {}
        super(AzureRMPrivateLinkService, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True, facts_module=False)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        old_response = self.get_item()
        result = None
        changed = False
        if not self.location:
            resource_group = self.get_resource_group(self.resource_group)
            self.location = resource_group.location
        self.body['location'] = self.location
        self.body['tags'] = self.tags
        if self.state == 'present':
            if old_response:
                update_tags, tags = self.update_tags(old_response['tags'])
                if update_tags:
                    changed = True
                self.body['tags'] = tags
                if self.body.get('enable_proxy_protocol') is not None:
                    if self.body.get('enable_proxy_protocol') != old_response['enable_proxy_protocol']:
                        changed = True
                else:
                    self.body['enable_proxy_protocol'] = old_response['enable_proxy_protocol']
                if self.body.get('auto_approval') is not None:
                    for value in old_response['auto_approval']['subscriptions']:
                        if value not in self.body['auto_approval']['subscriptions']:
                            self.body['auto_approval']['subscriptions'].append(value)
                    if len(self.body['auto_approval']['subscriptions']) != len(old_response['auto_approval']['subscriptions']):
                        changed = True
                else:
                    self.body['auto_approval'] = old_response['auto_approval']
                if self.body.get('visibility') is not None:
                    for value in old_response['visibility']['subscriptions']:
                        if value not in self.body['visibility']['subscriptions']:
                            self.body['visibility']['subscriptions'].append(value)
                    if len(self.body['visibility']['subscriptions']) != len(old_response['visibility']['subscriptions']):
                        changed = True
                else:
                    self.body['visibility'] = old_response['visibility']
                if self.body.get('fqdns') is not None:
                    for value in old_response['fqdns']:
                        if value not in self.body['fqdns']:
                            self.body['fqdns'].append(value)
                    if len(self.body.get('fqdns')) != len(old_response['fqdns']):
                        changed = True
                else:
                    self.body['fqdns'] = old_response['fqdns']
                if self.body.get('load_balancer_frontend_ip_configurations') is not None:
                    if self.body['load_balancer_frontend_ip_configurations'] != old_response['load_balancer_frontend_ip_configurations']:
                        self.fail('Private Link Service Load Balancer Reference Cannot Be Changed')
                else:
                    self.body['load_balancer_frontend_ip_configurations'] = old_response['load_balancer_frontend_ip_configurations']
                if self.body.get('ip_configurations') is not None:
                    for items in old_response['ip_configurations']:
                        if items['name'] not in [item['name'] for item in self.body['ip_configurations']]:
                            self.body['ip_configurations'].append(items)
                    if len(self.body['ip_configurations']) != len(old_response['ip_configurations']):
                        changed = True
                else:
                    self.body['ip_configurations'] = old_response['ip_configurations']
            else:
                changed = True
            if changed:
                if self.check_mode:
                    self.log('Check mode test. The private link service is exist, will be create or updated')
                else:
                    result = self.create_or_update(self.body)
            elif self.check_mode:
                self.log('Check mode test. The private endpoint connection is exist, No operation in this task')
            else:
                self.log('The private endpoint connection is exist, No operation in this task')
                result = old_response
        elif old_response:
            changed = True
            if self.check_mode:
                self.log('Check mode test. The private link service is exist, will be deleted')
            else:
                result = self.delete_resource()
        elif self.check_mode:
            self.log("The private link service isn't exist, no action")
        else:
            self.log("The private link service isn't exist, don't need to delete")
        self.results['link_service'] = result
        self.results['changed'] = changed
        return self.results

    def get_item(self):
        self.log('Get properties for {0} in {1}'.format(self.name, self.resource_group))
        try:
            response = self.network_client.private_link_services.get(self.resource_group, self.name)
            return self.service_to_dict(response)
        except ResourceNotFoundError:
            self.log('Could not get info for {0} in {1}'.format(self.name, self.resource_group))
        return []

    def create_or_update(self, parameters):
        self.log('Create or update the private link service for {0} in {1}'.format(self.name, self.resource_group))
        try:
            response = self.network_client.private_link_services.begin_create_or_update(self.resource_group, self.name, parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            result = self.network_client.private_link_services.get(self.resource_group, self.name)
            return self.service_to_dict(result)
        except Exception as ec:
            self.fail('Create or Update {0} in {1} failed, mesage {2}'.format(self.name, self.resource_group, ec))
        return []

    def delete_resource(self):
        self.log('delete the private link service for {0} in {1}'.format(self.name, self.resource_group))
        try:
            response = self.network_client.private_link_services.begin_delete(self.resource_group, self.name)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            return response
        except Exception as ec:
            self.fail('Delete {0} in {1} failed, message {2}'.format(self.name, self.resource_group, ec))
        return []

    def service_to_dict(self, service_info):
        service = service_info.as_dict()
        result = dict(id=service.get('id'), name=service.get('name'), type=service.get('type'), etag=service.get('etag'), location=service.get('location'), tags=service.get('tags'), load_balancer_frontend_ip_configurations=service.get('load_balancer_frontend_ip_configurations'), ip_configurations=list(), network_interfaces=service.get('network_interfaces'), provisioning_state=service.get('provisioning_state'), private_endpoint_connections=list(), visibility=service.get('visibility'), fqdns=service.get('fqdns'), auto_approval=service.get('auto_approval'), alias=service.get('alias'), enable_proxy_protocol=service.get('enable_proxy_protocol'))
        if service.get('private_endpoint_connections'):
            for items in service['private_endpoint_connections']:
                result['private_endpoint_connections'].append({'id': items['id'], 'private_endpoint': items['private_endpoint']['id']})
        if service.get('ip_configurations'):
            for items in service['ip_configurations']:
                result['ip_configurations'].append({'name': items['name'], 'properties': {'primary': items['primary'], 'private_ip_address_version': items['private_ip_address_version'], 'private_ip_allocation_method': items['private_ip_allocation_method'], 'subnet': items['subnet']}})
        return result