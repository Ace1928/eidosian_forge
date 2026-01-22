from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def antivirus_heuristic(data, fos):
    vdom = data['vdom']
    antivirus_heuristic_data = data['antivirus_heuristic']
    filtered_data = underscore_to_hyphen(filter_antivirus_heuristic_data(antivirus_heuristic_data))
    return fos.set('antivirus', 'heuristic', data=filtered_data, vdom=vdom)