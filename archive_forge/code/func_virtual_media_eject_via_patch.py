from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def virtual_media_eject_via_patch(self, uri, image_only=False):
    payload = {'Inserted': False, 'Image': None}
    if image_only:
        del payload['Inserted']
    resp = self.patch_request(self.root_uri + uri, payload, check_pyld=True)
    if resp['ret'] is False and 'Inserted' in payload:
        vendor = self._get_vendor()['Vendor']
        if vendor == 'HPE' or vendor == 'Supermicro':
            payload.pop('Inserted', None)
            resp = self.patch_request(self.root_uri + uri, payload, check_pyld=True)
    if resp['ret'] and resp['changed']:
        resp['msg'] = 'VirtualMedia ejected'
    return resp