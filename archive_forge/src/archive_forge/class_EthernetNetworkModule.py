from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleResourceNotFound
class EthernetNetworkModule(OneViewModuleBase):
    MSG_CREATED = 'Ethernet Network created successfully.'
    MSG_UPDATED = 'Ethernet Network updated successfully.'
    MSG_DELETED = 'Ethernet Network deleted successfully.'
    MSG_ALREADY_PRESENT = 'Ethernet Network is already present.'
    MSG_ALREADY_ABSENT = 'Ethernet Network is already absent.'
    MSG_BULK_CREATED = 'Ethernet Networks created successfully.'
    MSG_MISSING_BULK_CREATED = 'Some missing Ethernet Networks were created successfully.'
    MSG_BULK_ALREADY_EXIST = 'The specified Ethernet Networks already exist.'
    MSG_CONNECTION_TEMPLATE_RESET = 'Ethernet Network connection template was reset to the default.'
    MSG_ETHERNET_NETWORK_NOT_FOUND = 'Ethernet Network was not found.'
    RESOURCE_FACT_NAME = 'ethernet_network'

    def __init__(self):
        argument_spec = dict(state=dict(type='str', default='present', choices=['absent', 'default_bandwidth_reset', 'present']), data=dict(type='dict', required=True))
        super(EthernetNetworkModule, self).__init__(additional_arg_spec=argument_spec, validate_etag_support=True)
        self.resource_client = self.oneview_client.ethernet_networks

    def execute_module(self):
        changed, msg, ansible_facts, resource = (False, '', {}, None)
        if self.data.get('name'):
            resource = self.get_by_name(self.data['name'])
        if self.state == 'present':
            if self.data.get('vlanIdRange'):
                return self._bulk_present()
            else:
                return self._present(resource)
        elif self.state == 'absent':
            return self.resource_absent(resource)
        elif self.state == 'default_bandwidth_reset':
            changed, msg, ansible_facts = self._default_bandwidth_reset(resource)
            return dict(changed=changed, msg=msg, ansible_facts=ansible_facts)

    def _present(self, resource):
        bandwidth = self.data.pop('bandwidth', None)
        scope_uris = self.data.pop('scopeUris', None)
        result = self.resource_present(resource, self.RESOURCE_FACT_NAME)
        if bandwidth:
            if self._update_connection_template(result['ansible_facts']['ethernet_network'], bandwidth)[0]:
                result['changed'] = True
                result['msg'] = self.MSG_UPDATED
        if scope_uris is not None:
            result = self.resource_scopes_set(result, 'ethernet_network', scope_uris)
        return result

    def _bulk_present(self):
        vlan_id_range = self.data['vlanIdRange']
        result = dict(ansible_facts={})
        ethernet_networks = self.resource_client.get_range(self.data['namePrefix'], vlan_id_range)
        if not ethernet_networks:
            self.resource_client.create_bulk(self.data)
            result['changed'] = True
            result['msg'] = self.MSG_BULK_CREATED
        else:
            vlan_ids = self.resource_client.dissociate_values_or_ranges(vlan_id_range)
            for net in ethernet_networks[:]:
                vlan_ids.remove(net['vlanId'])
            if len(vlan_ids) == 0:
                result['msg'] = self.MSG_BULK_ALREADY_EXIST
                result['changed'] = False
            else:
                if len(vlan_ids) == 1:
                    self.data['vlanIdRange'] = '{0}-{1}'.format(vlan_ids[0], vlan_ids[0])
                else:
                    self.data['vlanIdRange'] = ','.join(map(str, vlan_ids))
                self.resource_client.create_bulk(self.data)
                result['changed'] = True
                result['msg'] = self.MSG_MISSING_BULK_CREATED
        result['ansible_facts']['ethernet_network_bulk'] = self.resource_client.get_range(self.data['namePrefix'], vlan_id_range)
        return result

    def _update_connection_template(self, ethernet_network, bandwidth):
        if 'connectionTemplateUri' not in ethernet_network:
            return (False, None)
        connection_template = self.oneview_client.connection_templates.get(ethernet_network['connectionTemplateUri'])
        merged_data = connection_template.copy()
        merged_data.update({'bandwidth': bandwidth})
        if not self.compare(connection_template, merged_data):
            connection_template = self.oneview_client.connection_templates.update(merged_data)
            return (True, connection_template)
        else:
            return (False, None)

    def _default_bandwidth_reset(self, resource):
        if not resource:
            raise OneViewModuleResourceNotFound(self.MSG_ETHERNET_NETWORK_NOT_FOUND)
        default_connection_template = self.oneview_client.connection_templates.get_default()
        changed, connection_template = self._update_connection_template(resource, default_connection_template['bandwidth'])
        return (changed, self.MSG_CONNECTION_TEMPLATE_RESET, dict(ethernet_network_connection_template=connection_template))