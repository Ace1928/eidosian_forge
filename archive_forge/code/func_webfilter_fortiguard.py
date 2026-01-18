from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def webfilter_fortiguard(data, fos):
    vdom = data['vdom']
    webfilter_fortiguard_data = data['webfilter_fortiguard']
    filtered_data = underscore_to_hyphen(filter_webfilter_fortiguard_data(webfilter_fortiguard_data))
    return fos.set('webfilter', 'fortiguard', data=filtered_data, vdom=vdom)