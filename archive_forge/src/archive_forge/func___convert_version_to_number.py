from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def __convert_version_to_number(version):
    version = version[1:] if version.startswith('v') else version
    seg = version.split('.')
    if len(seg) != 3:
        raise 'Invalid fortios system version number: ' + version + '. Should be of format [major].[minor].[patch]'
    return int(seg[0]) * 10000 + int(seg[1]) * 100 + int(seg[2])