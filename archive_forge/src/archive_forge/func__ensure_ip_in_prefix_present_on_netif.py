from __future__ import absolute_import, division, print_function
from ipaddress import ip_interface
from ansible.module_utils._text import to_text
from ansible_collections.netbox.netbox.plugins.module_utils.netbox_utils import (
def _ensure_ip_in_prefix_present_on_netif(self, nb_app, nb_endpoint, data, endpoint_name):
    query_params = {'parent': data['prefix']}
    if not self._version_check_greater(self.version, '2.9', greater_or_equal=True):
        if not data.get('interface') or not data.get('prefix'):
            self._handle_errors('A prefix and interface is required')
        data_intf_key = 'interface'
    else:
        if not data.get('assigned_object_id') or not data.get('prefix'):
            self._handle_errors('A prefix and assigned_object is required')
        data_intf_key = 'assigned_object_id'
    intf_obj_type = data.get('assigned_object_type', 'dcim.interface')
    if intf_obj_type == 'virtualization.vminterface':
        intf_type = 'vminterface_id'
    else:
        intf_type = 'interface_id'
    query_params.update({intf_type: data[data_intf_key]})
    if data.get('vrf'):
        query_params['vrf_id'] = data['vrf']
    attached_ips = nb_endpoint.filter(**query_params)
    if attached_ips:
        self.nb_object = list(attached_ips)[-1].serialize()
        self.result['changed'] = False
        self.result['msg'] = '%s %s already attached' % (endpoint_name, self.nb_object['address'])
    else:
        self._get_new_available_ip_address(nb_app, data, endpoint_name)