from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def antivirus_profile(data, fos):
    vdom = data['vdom']
    state = data['state']
    antivirus_profile_data = data['antivirus_profile']
    antivirus_profile_data = flatten_multilists_attributes(antivirus_profile_data)
    filtered_data = underscore_to_hyphen(filter_antivirus_profile_data(antivirus_profile_data))
    if state == 'present' or state is True:
        return fos.set('antivirus', 'profile', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('antivirus', 'profile', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')