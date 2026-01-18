from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def wireless_controller_setting(data, fos):
    vdom = data['vdom']
    wireless_controller_setting_data = data['wireless_controller_setting']
    wireless_controller_setting_data = flatten_multilists_attributes(wireless_controller_setting_data)
    filtered_data = underscore_to_hyphen(filter_wireless_controller_setting_data(wireless_controller_setting_data))
    return fos.set('wireless-controller', 'setting', data=filtered_data, vdom=vdom)