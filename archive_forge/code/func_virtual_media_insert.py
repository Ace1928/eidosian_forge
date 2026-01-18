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
def virtual_media_insert(self, options, resource_type='Manager'):
    param_map = {'Inserted': 'inserted', 'WriteProtected': 'write_protected', 'UserName': 'username', 'Password': 'password', 'TransferProtocolType': 'transfer_protocol_type', 'TransferMethod': 'transfer_method'}
    image_url = options.get('image_url')
    if not image_url:
        return {'ret': False, 'msg': 'image_url option required for VirtualMediaInsert'}
    media_types = options.get('media_types')
    if resource_type == 'Systems':
        resource_uri = self.systems_uri
    elif resource_type == 'Manager':
        resource_uri = self.manager_uri
    response = self.get_request(self.root_uri + resource_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'VirtualMedia' not in data:
        return {'ret': False, 'msg': 'VirtualMedia resource not found'}
    virt_media_uri = data['VirtualMedia']['@odata.id']
    response = self.get_request(self.root_uri + virt_media_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    virt_media_list = []
    for member in data[u'Members']:
        virt_media_list.append(member[u'@odata.id'])
    resources, headers = self._read_virt_media_resources(virt_media_list)
    if self._virt_media_image_inserted(resources, image_url):
        return {'ret': True, 'changed': False, 'msg': "VirtualMedia '%s' already inserted" % image_url}
    vendor = self._get_vendor()['Vendor']
    uri, data = self._find_empty_virt_media_slot(resources, media_types, media_match_strict=True, vendor=vendor)
    if not uri:
        uri, data = self._find_empty_virt_media_slot(resources, media_types, media_match_strict=False, vendor=vendor)
    if not uri:
        return {'ret': False, 'msg': 'Unable to find an available VirtualMedia resource %s' % ('supporting ' + str(media_types) if media_types else '')}
    if 'Actions' not in data or '#VirtualMedia.InsertMedia' not in data['Actions']:
        h = headers[uri]
        if 'allow' in h:
            methods = [m.strip() for m in h.get('allow').split(',')]
            if 'PATCH' not in methods:
                return {'ret': False, 'msg': '%s action not found and PATCH not allowed' % '#VirtualMedia.InsertMedia'}
        return self.virtual_media_insert_via_patch(options, param_map, uri, data)
    action = data['Actions']['#VirtualMedia.InsertMedia']
    if 'target' not in action:
        return {'ret': False, 'msg': 'target URI missing from Action #VirtualMedia.InsertMedia'}
    action_uri = action['target']
    ai = self._get_all_action_info_values(action)
    payload = self._insert_virt_media_payload(options, param_map, data, ai)
    response = self.post_request(self.root_uri + action_uri, payload)
    if response['ret'] is False and ('Inserted' in payload or 'WriteProtected' in payload):
        vendor = self._get_vendor()['Vendor']
        if vendor == 'HPE' or vendor == 'Supermicro':
            payload.pop('Inserted', None)
            payload.pop('WriteProtected', None)
            response = self.post_request(self.root_uri + action_uri, payload)
    if response['ret'] is False:
        return response
    return {'ret': True, 'changed': True, 'msg': 'VirtualMedia inserted'}