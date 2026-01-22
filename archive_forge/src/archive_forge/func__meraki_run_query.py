from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
from re import sub
def _meraki_run_query(meraki):
    """Get the radio settings on the specified device."""
    meraki.result['data'] = meraki_get_radio_settings(meraki)
    meraki.exit_json(**meraki.result)