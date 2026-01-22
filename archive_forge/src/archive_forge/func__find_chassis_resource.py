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
def _find_chassis_resource(self):
    response = self.get_request(self.root_uri + self.service_root)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'Chassis' not in data:
        return {'ret': False, 'msg': 'Chassis resource not found'}
    chassis = data['Chassis']['@odata.id']
    response = self.get_request(self.root_uri + chassis)
    if response['ret'] is False:
        return response
    self.chassis_uris = [i['@odata.id'] for i in response['data'].get('Members', [])]
    if not self.chassis_uris:
        return {'ret': False, 'msg': 'Chassis Members array is either empty or missing'}
    self.chassis_uri = self.chassis_uris[0]
    if self.data_modification:
        if self.resource_id:
            self.chassis_uri = self._get_resource_uri_by_id(self.chassis_uris, self.resource_id)
            if not self.chassis_uri:
                return {'ret': False, 'msg': 'Chassis resource %s not found' % self.resource_id}
        elif len(self.chassis_uris) > 1:
            self.module.fail_json(msg=FAIL_MSG % {'resource': 'Chassis'})
    return {'ret': True}