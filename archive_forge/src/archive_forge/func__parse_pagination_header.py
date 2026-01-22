from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
@staticmethod
def _parse_pagination_header(link):
    rels = {'first': None, 'next': None, 'prev': None, 'last': None}
    for rel in link.split(','):
        kv = rel.split('rel=')
        rels[kv[1]] = kv[0].split('<')[1].split('>')[0].strip()
    return rels