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
def verify_bios_attributes(self, bios_attributes):
    server_bios = self.get_multi_bios_attributes()
    if server_bios['ret'] is False:
        return server_bios
    bios_dict = {}
    wrong_param = {}
    for key, value in bios_attributes.items():
        if key in server_bios['entries'][0][1]:
            if server_bios['entries'][0][1][key] != value:
                bios_dict.update({key: value})
        else:
            wrong_param.update({key: value})
    if wrong_param:
        return {'ret': False, 'msg': 'Wrong parameters are provided: %s' % wrong_param}
    if bios_dict:
        return {'ret': False, 'msg': 'BIOS parameters are not matching: %s' % bios_dict}
    return {'ret': True, 'changed': False, 'msg': 'BIOS verification completed'}