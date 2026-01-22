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
@staticmethod
def _insert_virt_media_payload(options, param_map, data, ai):
    payload = {'Image': options.get('image_url')}
    for param, option in param_map.items():
        if options.get(option) is not None and param in data:
            allowable = ai.get(param, {}).get('AllowableValues', [])
            if allowable and options.get(option) not in allowable:
                return {'ret': False, 'msg': "Value '%s' specified for option '%s' not in list of AllowableValues %s" % (options.get(option), option, allowable)}
            payload[param] = options.get(option)
    return payload