from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def antivirus_exempt_list(data, fos):
    vdom = data['vdom']
    state = data['state']
    antivirus_exempt_list_data = data['antivirus_exempt_list']
    filtered_data = underscore_to_hyphen(filter_antivirus_exempt_list_data(antivirus_exempt_list_data))
    if state == 'present' or state is True:
        return fos.set('antivirus', 'exempt-list', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('antivirus', 'exempt-list', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')