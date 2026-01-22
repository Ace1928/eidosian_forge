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
def _find_managers_resource(self):
    response = self.get_request(self.root_uri + self.service_root)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'Managers' not in data:
        return {'ret': False, 'msg': 'Manager resource not found'}
    manager = data['Managers']['@odata.id']
    response = self.get_request(self.root_uri + manager)
    if response['ret'] is False:
        return response
    self.manager_uris = [i['@odata.id'] for i in response['data'].get('Members', [])]
    if not self.manager_uris:
        return {'ret': False, 'msg': 'Managers Members array is either empty or missing'}
    self.manager_uri = self.manager_uris[0]
    if self.data_modification:
        if self.resource_id:
            self.manager_uri = self._get_resource_uri_by_id(self.manager_uris, self.resource_id)
            if not self.manager_uri:
                return {'ret': False, 'msg': 'Manager resource %s not found' % self.resource_id}
        elif len(self.manager_uris) > 1:
            self.module.fail_json(msg=FAIL_MSG % {'resource': 'Manager'})
    return {'ret': True}