from __future__ import absolute_import, division, print_function
import json
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.aci.plugins.module_utils.aci import ACIModule, aci_argument_spec
from ansible.module_utils._text import to_text
def update_qsl(url, params):
    """Add or update a URL query string"""
    if HAS_URLPARSE:
        url_parts = list(urlparse(url))
        query = dict(parse_qsl(url_parts[4]))
        query.update(params)
        url_parts[4] = urlencode(query)
        return urlunparse(url_parts)
    elif '?' in url:
        return url + '&' + '&'.join(['%s=%s' % (k, v) for k, v in params.items()])
    else:
        return url + '?' + '&'.join(['%s=%s' % (k, v) for k, v in params.items()])