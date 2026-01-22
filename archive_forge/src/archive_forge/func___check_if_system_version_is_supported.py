from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def __check_if_system_version_is_supported(v_range, version):
    """check the current system version is supported in v_range"""
    if not v_range:
        return {'supported': True}
    system_version_number = __convert_version_to_number(version)
    v_range_desc = ', '.join(list(map(__format_single_range_desc, v_range)))
    for [single_range_start, single_range_end] in v_range:
        single_range_start_number = __convert_version_to_number(single_range_start)
        if system_version_number < single_range_start_number:
            return {'supported': False, 'reason': 'Supported version ranges are ' + v_range_desc}
        if single_range_end == '' or system_version_number <= __convert_version_to_number(single_range_end):
            return {'supported': True}
    return {'supported': False, 'reason': 'Supported version ranges are ' + v_range_desc}