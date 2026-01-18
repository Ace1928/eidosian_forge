from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat, pprint
import time
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def update_api_address_interface_match(self, body):
    """Change network interface address which matches the api_address"""
    try:
        try:
            rc, data = request(self.url + 'storage-systems/%s/configuration/ethernet-interfaces' % self.ssid, use_proxy=False, force=True, ignore_errors=True, method='POST', data=json.dumps(body), headers=HEADERS, timeout=10, **self.creds)
        except Exception:
            url_parts = list(urlparse.urlparse(self.url))
            domain = url_parts[1].split(':')
            domain[0] = self.address
            url_parts[1] = ':'.join(domain)
            expected_url = urlparse.urlunparse(url_parts)
            self._logger.info(pformat(expected_url))
            rc, data = request(expected_url + 'storage-systems/%s/configuration/ethernet-interfaces' % self.ssid, headers=HEADERS, timeout=300, **self.creds)
            return
    except Exception as err:
        self._logger.info(type(err))
        self.module.fail_json(msg='Connection failure: we failed to modify the network settings! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))