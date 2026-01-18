from __future__ import absolute_import, division, print_function
import json
import multiprocessing
import threading
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import request
from ansible.module_utils._text import to_native
def test_systems_found(self, systems_found, serial, label, addresses):
    """Verify and build api urls."""
    api_urls = []
    for address in addresses:
        for port in self.ports:
            if port == '8080':
                url = 'http://%s:%s/devmgr/' % (address, port)
            else:
                url = 'https://%s:%s/devmgr/' % (address, port)
            try:
                rc, response = request(url + 'utils/about', validate_certs=False, timeout=self.SEARCH_TIMEOUT)
                api_urls.append(url + 'v2/')
                break
            except Exception as error:
                pass
    systems_found.update({serial: {'api_urls': api_urls, 'label': label, 'addresses': addresses, 'proxy_ssid': '', 'proxy_required': False}})