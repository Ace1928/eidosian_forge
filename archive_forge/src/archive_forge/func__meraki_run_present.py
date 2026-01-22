from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
from re import sub
def _meraki_run_present(meraki):
    """Update / check radio settings for a specified device."""
    original = meraki_get_radio_settings(meraki)
    meraki.result['data'] = original
    meraki.params['rf_profile_id'] = get_rf_profile_id(meraki)
    payload = construct_payload(meraki)
    if meraki.is_update_required(original, payload) is True:
        if meraki.check_mode is True:
            meraki.result['data'] = payload
            meraki.result['changed'] = True
            meraki.result['original'] = original
        else:
            path = meraki.construct_path('update', custom={'serial': meraki.params['serial']})
            response = meraki.request(path, method='PUT', payload=json.dumps(payload))
            meraki.result['data'] = response
            meraki.result['changed'] = True
    meraki.exit_json(**meraki.result)