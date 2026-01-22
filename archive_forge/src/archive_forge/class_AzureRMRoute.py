from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _snake_to_camel
class AzureRMRoute(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), address_prefix=dict(type='str'), next_hop_type=dict(type='str', choices=['virtual_network_gateway', 'vnet_local', 'internet', 'virtual_appliance', 'none'], default='none'), next_hop_ip_address=dict(type='str'), route_table_name=dict(type='str', required=True))
        required_if = [('state', 'present', ['next_hop_type'])]
        self.resource_group = None
        self.name = None
        self.state = None
        self.address_prefix = None
        self.next_hop_type = None
        self.next_hop_ip_address = None
        self.route_table_name = None
        self.results = dict(changed=False, id=None)
        super(AzureRMRoute, self).__init__(self.module_arg_spec, required_if=required_if, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        result = dict()
        changed = False
        self.next_hop_type = _snake_to_camel(self.next_hop_type, capitalize_first=True)
        result = self.get_route()
        if self.state == 'absent' and result:
            changed = True
            if not self.check_mode:
                self.delete_route()
        elif self.state == 'present':
            if not result:
                changed = True
            else:
                if result.next_hop_type != self.next_hop_type:
                    self.log('Update: {0} next_hop_type from {1} to {2}'.format(self.name, result.next_hop_type, self.next_hop_type))
                    changed = True
                if result.next_hop_ip_address != self.next_hop_ip_address:
                    self.log('Update: {0} next_hop_ip_address from {1} to {2}'.format(self.name, result.next_hop_ip_address, self.next_hop_ip_address))
                    changed = True
                if result.address_prefix != self.address_prefix:
                    self.log('Update: {0} address_prefix from {1} to {2}'.format(self.name, result.address_prefix, self.address_prefix))
                    changed = True
            if changed:
                result = self.network_models.Route(name=self.name, address_prefix=self.address_prefix, next_hop_type=self.next_hop_type, next_hop_ip_address=self.next_hop_ip_address)
                if not self.check_mode:
                    result = self.create_or_update_route(result)
        self.results['id'] = result.id if result else None
        self.results['changed'] = changed
        return self.results

    def create_or_update_route(self, param):
        try:
            poller = self.network_client.routes.begin_create_or_update(self.resource_group, self.route_table_name, self.name, param)
            return self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error creating or updating route {0} - {1}'.format(self.name, str(exc)))

    def delete_route(self):
        try:
            poller = self.network_client.routes.begin_delete(self.resource_group, self.route_table_name, self.name)
            result = self.get_poller_result(poller)
            return result
        except Exception as exc:
            self.fail('Error deleting route {0} - {1}'.format(self.name, str(exc)))

    def get_route(self):
        try:
            return self.network_client.routes.get(self.resource_group, self.route_table_name, self.name)
        except ResourceNotFoundError as cloud_err:
            if cloud_err.status_code == 404:
                self.log('{0}'.format(str(cloud_err)))
                return None
            self.fail('Error: failed to get resource {0} - {1}'.format(self.name, str(cloud_err)))
        except Exception as exc:
            self.fail('Error: failed to get resource {0} - {1}'.format(self.name, str(exc)))