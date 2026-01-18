from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
from re import sub
def update_url_catalog(meraki):
    """Update the URL catalog available to the helper."""
    query_urls = {'mr_radio': '/devices/{serial}/wireless/radio/settings'}
    update_urls = {'mr_radio': '/devices/{serial}/wireless/radio/settings'}
    query_all_urls = {'mr_rf_profile': '/networks/{net_id}/wireless/rfProfiles'}
    meraki.url_catalog['get_one'].update(query_urls)
    meraki.url_catalog['update'] = update_urls
    meraki.url_catalog['get_all'].update(query_all_urls)