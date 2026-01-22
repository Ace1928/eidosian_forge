from __future__ import absolute_import, division, print_function
import json
import os
import uuid
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def activate_firmware_image(self):
    """Perform a Firmware Activate on the OCAPI storage device."""
    resource_uri = self.root_uri
    resource_uri = self.get_uri_with_slot_number_query_param(resource_uri)
    response = self.get_request(resource_uri)
    if 'etag' not in response['headers']:
        return {'ret': False, 'msg': 'Etag not found in response.'}
    etag = response['headers']['etag']
    if response['ret'] is False:
        return response
    if self.module.check_mode:
        return {'ret': True, 'changed': True, 'msg': 'Update not performed in check mode.'}
    payload = {'FirmwareActivate': True}
    response = self.put_request(resource_uri, payload, etag)
    if response['ret'] is False:
        return response
    return {'ret': True, 'jobUri': response['headers']['location']}