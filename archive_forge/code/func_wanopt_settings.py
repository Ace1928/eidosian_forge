from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def wanopt_settings(data, fos):
    vdom = data['vdom']
    wanopt_settings_data = data['wanopt_settings']
    filtered_data = underscore_to_hyphen(filter_wanopt_settings_data(wanopt_settings_data))
    return fos.set('wanopt', 'settings', data=filtered_data, vdom=vdom)