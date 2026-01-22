from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class FcoeNetworkModule(OneViewModuleBase):
    MSG_CREATED = 'FCoE Network created successfully.'
    MSG_UPDATED = 'FCoE Network updated successfully.'
    MSG_DELETED = 'FCoE Network deleted successfully.'
    MSG_ALREADY_PRESENT = 'FCoE Network is already present.'
    MSG_ALREADY_ABSENT = 'FCoE Network is already absent.'
    RESOURCE_FACT_NAME = 'fcoe_network'

    def __init__(self):
        additional_arg_spec = dict(data=dict(required=True, type='dict'), state=dict(default='present', choices=['present', 'absent']))
        super(FcoeNetworkModule, self).__init__(additional_arg_spec=additional_arg_spec, validate_etag_support=True)
        self.resource_client = self.oneview_client.fcoe_networks

    def execute_module(self):
        resource = self.get_by_name(self.data.get('name'))
        if self.state == 'present':
            return self.__present(resource)
        elif self.state == 'absent':
            return self.resource_absent(resource)

    def __present(self, resource):
        scope_uris = self.data.pop('scopeUris', None)
        result = self.resource_present(resource, self.RESOURCE_FACT_NAME)
        if scope_uris is not None:
            result = self.resource_scopes_set(result, 'fcoe_network', scope_uris)
        return result