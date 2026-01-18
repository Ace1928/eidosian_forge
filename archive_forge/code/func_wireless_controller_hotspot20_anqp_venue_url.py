from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def wireless_controller_hotspot20_anqp_venue_url(data, fos):
    vdom = data['vdom']
    state = data['state']
    wireless_controller_hotspot20_anqp_venue_url_data = data['wireless_controller_hotspot20_anqp_venue_url']
    filtered_data = underscore_to_hyphen(filter_wireless_controller_hotspot20_anqp_venue_url_data(wireless_controller_hotspot20_anqp_venue_url_data))
    if state == 'present' or state is True:
        return fos.set('wireless-controller.hotspot20', 'anqp-venue-url', data=filtered_data, vdom=vdom)
    elif state == 'absent':
        return fos.delete('wireless-controller.hotspot20', 'anqp-venue-url', mkey=filtered_data['name'], vdom=vdom)
    else:
        fos._module.fail_json(msg='state must be present or absent!')