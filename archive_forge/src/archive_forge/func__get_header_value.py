from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def _get_header_value(info, header_name):
    header_name = header_name.lower()
    header_value = info.get(header_name)
    for k, v in info.items():
        if k.lower() == header_name:
            header_value = v
    return header_value