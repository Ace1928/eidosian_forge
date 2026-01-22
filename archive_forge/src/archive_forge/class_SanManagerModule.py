from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleValueError
class SanManagerModule(OneViewModuleBase):
    MSG_CREATED = 'SAN Manager created successfully.'
    MSG_UPDATED = 'SAN Manager updated successfully.'
    MSG_DELETED = 'SAN Manager deleted successfully.'
    MSG_ALREADY_PRESENT = 'SAN Manager is already present.'
    MSG_ALREADY_ABSENT = 'SAN Manager is already absent.'
    MSG_SAN_MANAGER_PROVIDER_DISPLAY_NAME_NOT_FOUND = "The provider '{0}' was not found."
    argument_spec = dict(state=dict(type='str', default='present', choices=['absent', 'present', 'connection_information_set']), data=dict(type='dict', required=True))

    def __init__(self):
        super(SanManagerModule, self).__init__(additional_arg_spec=self.argument_spec, validate_etag_support=True)
        self.resource_client = self.oneview_client.san_managers

    def execute_module(self):
        if self.data.get('connectionInfo'):
            for connection_hash in self.data.get('connectionInfo'):
                if connection_hash.get('name') == 'Host':
                    resource_name = connection_hash.get('value')
        elif self.data.get('name'):
            resource_name = self.data.get('name')
        else:
            msg = 'A "name" or "connectionInfo" must be provided inside the "data" field for this operation. '
            msg += 'If a "connectionInfo" is provided, the "Host" name is considered as the "name" for the resource.'
            raise OneViewModuleValueError(msg.format())
        resource = self.resource_client.get_by_name(resource_name)
        if self.state == 'present':
            changed, msg, san_manager = self._present(resource)
            return dict(changed=changed, msg=msg, ansible_facts=dict(san_manager=san_manager))
        elif self.state == 'absent':
            return self.resource_absent(resource, method='remove')
        elif self.state == 'connection_information_set':
            changed, msg, san_manager = self._connection_information_set(resource)
            return dict(changed=changed, msg=msg, ansible_facts=dict(san_manager=san_manager))

    def _present(self, resource):
        if not resource:
            provider_uri = self.data.get('providerUri', self._get_provider_uri_by_display_name(self.data))
            return (True, self.MSG_CREATED, self.resource_client.add(self.data, provider_uri))
        else:
            merged_data = resource.copy()
            merged_data.update(self.data)
            resource.pop('connectionInfo', None)
            merged_data.pop('connectionInfo', None)
            if self.compare(resource, merged_data):
                return (False, self.MSG_ALREADY_PRESENT, resource)
            else:
                updated_san_manager = self.resource_client.update(resource=merged_data, id_or_uri=resource['uri'])
                return (True, self.MSG_UPDATED, updated_san_manager)

    def _connection_information_set(self, resource):
        if not resource:
            return self._present(resource)
        else:
            merged_data = resource.copy()
            merged_data.update(self.data)
            merged_data.pop('refreshState', None)
            if not self.data.get('connectionInfo', None):
                raise OneViewModuleValueError('A connectionInfo field is required for this operation.')
            updated_san_manager = self.resource_client.update(resource=merged_data, id_or_uri=resource['uri'])
            return (True, self.MSG_UPDATED, updated_san_manager)

    def _get_provider_uri_by_display_name(self, data):
        display_name = data.get('providerDisplayName')
        provider_uri = self.resource_client.get_provider_uri(display_name)
        if not provider_uri:
            raise OneViewModuleValueError(self.MSG_SAN_MANAGER_PROVIDER_DISPLAY_NAME_NOT_FOUND.format(display_name))
        return provider_uri