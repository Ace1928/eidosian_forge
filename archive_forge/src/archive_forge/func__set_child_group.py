from __future__ import (absolute_import, division, print_function)
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
def _set_child_group(self, group_data):
    port = self.get_option('port') if 'port' in self.config else 443
    validate_certs = self.get_option('validate_certs') if 'validate_certs' in self.config else False
    module_params = {'hostname': self.get_option('hostname'), 'username': self.get_option('username'), 'password': self.get_option('password'), 'port': port, 'validate_certs': validate_certs}
    if 'ca_path' in self.config:
        module_params.update({'ca_path': self.get_option('ca_path')})
    with RestOME(module_params, req_session=False) as ome:
        for gdata in group_data:
            group_name = gdata['Name']
            subgroup_uri = gdata['SubGroups@odata.navigationLink'].strip('/api/')
            sub_group = get_all_data_with_pagination(ome, subgroup_uri)
            gdata = sub_group.get('report_list', [])
            if gdata:
                self._add_group_data(gdata)
                self._add_child_group_data(group_name, gdata)